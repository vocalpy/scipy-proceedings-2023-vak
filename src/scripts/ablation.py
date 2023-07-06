from __future__ import annotations

import torch
from torch import nn

import vak
from vak import (
    metrics,
    nets
)
from vak.models.frame_classification_model import FrameClassificationModel
from vak.models.decorator import model
from vak.nn.modules import Conv2dTF


class TweetyNetNoLSTM(nn.Module):
    """TweetyNet model without bidirectional LSTM layer."""
    def __init__(self,
                 num_classes,
                 num_input_channels=1,
                 num_freqbins=256,
                 padding='SAME',
                 conv1_filters=32,
                 conv1_kernel_size=(5, 5),
                 conv2_filters=64,
                 conv2_kernel_size=(5, 5),
                 pool1_size=(8, 1),
                 pool1_stride=(8, 1),
                 pool2_size=(8, 1),
                 pool2_stride=(8, 1),
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.num_freqbins = num_freqbins

        self.cnn = nn.Sequential(
            Conv2dTF(in_channels=self.num_input_channels,
                     out_channels=conv1_filters,
                     kernel_size=conv1_kernel_size,
                     padding=padding
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool1_size,
                         stride=pool1_stride),
            Conv2dTF(in_channels=conv1_filters,
                     out_channels=conv2_filters,
                     kernel_size=conv2_kernel_size,
                     padding=padding,
                     ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool2_size,
                         stride=pool2_stride),
        )

        # determine number of features in output after stacking channels
        # we use the same number of features for hidden states
        # note self.num_hidden is also used to reshape output of cnn in self.forward method
        N_DUMMY_TIMEBINS = 256  # some not-small number. This dimension doesn't matter here
        batch_shape = (1, self.num_input_channels, self.num_freqbins, N_DUMMY_TIMEBINS)
        tmp_tensor = torch.rand(batch_shape)
        tmp_out = self.cnn(tmp_tensor)
        channels_out, freqbins_out = tmp_out.shape[1], tmp_out.shape[2]
        self.cnn_out_features = channels_out * freqbins_out

        self.fc = nn.Linear(in_features=self.cnn_out_features, out_features=num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # stack channels, to give tensor shape (batch, cnn_out_features, num time bins)
        features = features.view(features.shape[0], self.cnn_out_features, -1)
        # permute to (batch, time bins, hidden size) to project features down onto number of classes
        features = features.permute(0, 2, 1)
        logits = self.fc(features)
        # permute again so that dimension order is (batch, classes, time steps)
        # because this is order that loss function expects
        return logits.permute(0, 2, 1)


@model(family=FrameClassificationModel)
class TweetyNoLSTMNet:
    """TweetyNet model without LSTM layer"""
    network = TweetyNetNoLSTM
    loss = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    metrics = {'acc': metrics.Accuracy,
               'levenshtein': metrics.Levenshtein,
               'segment_error_rate': metrics.SegmentErrorRate,
               'loss': torch.nn.CrossEntropyLoss}
    default_config = {
        'optimizer':
            {'lr': 0.003}
    }


if __name__ == '__main__':
    parser = vak.__main__.get_parser()
    args = parser.parse_args()
    vak.cli.cli.cli(command=args.command, config_file=args.configfile)
