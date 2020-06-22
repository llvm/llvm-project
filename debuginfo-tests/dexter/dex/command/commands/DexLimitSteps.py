# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Command that enables test writers to specify a limited number of break
points using an start condition and range.
"""

from dex.command.CommandBase import CommandBase

class DexLimitSteps(CommandBase):
    def __init__(self, *args, **kwargs):
        self.expression = args[0]
        self.values = [str(arg) for arg in args[1:]]
        try:
            on_line = kwargs.pop('on_line')
            self.from_line = on_line
            self.to_line = on_line
        except KeyError:
            self.from_line = kwargs.pop('from_line', 1)
            self.to_line = kwargs.pop('to_line', 999999)
        if kwargs:
            raise TypeError('unexpected named args: {}'.format(
                ', '.join(kwargs)))
        super(DexLimitSteps, self).__init__()

    def resolve_label(self, label_line_pair):
        label, lineno = label_line_pair
        if isinstance(self.from_line, str):
            if self.from_line == label:
                self.from_line = lineno
        if isinstance(self.to_line, str):
            if self.to_line == label:
                self.to_line = lineno

    def has_labels(self):
        return len(self.get_label_args()) > 0

    def get_label_args(self):
        return [label for label in (self.from_line, self.to_line)
                      if isinstance(label, str)]

    def eval(self):
        raise NotImplementedError('DexLimitSteps commands cannot be evaled.')

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
