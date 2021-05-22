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
        if len(args) == 0:
            self.expression = None
            self.values = []
        elif len(args) == 1:
            raise TypeError("expected 0 or at least 2 positional arguments")
        else:
            self.expression = args[0]
            self.values = [str(arg) for arg in args[1:]]
        try:
            on_line = kwargs.pop('on_line')
            self.from_line = on_line
            self.to_line = on_line
        except KeyError:
            self.from_line = kwargs.pop('from_line', 1)
            self.to_line = kwargs.pop('to_line', 999999)
        self.hit_count = kwargs.pop('hit_count', None)
        if kwargs:
            raise TypeError('unexpected named args: {}'.format(
                ', '.join(kwargs)))
        super(DexLimitSteps, self).__init__()

    def eval(self):
        raise NotImplementedError('DexLimitSteps commands cannot be evaled.')

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
