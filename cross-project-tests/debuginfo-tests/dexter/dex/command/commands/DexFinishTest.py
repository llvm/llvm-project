# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Command that enables test writers to terminate a test after a specified
breakpoint has been hit a number of times.
"""

from dex.command.CommandBase import CommandBase

class DexFinishTest(CommandBase):
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self.expression = None
            self.values = []
        elif len(args) == 1:
            raise TypeError("expected 0 or at least 2 positional arguments")
        else:
            self.expression = args[0]
            self.values = [str(arg) for arg in args[1:]]
        self.on_line = kwargs.pop('on_line')
        self.hit_count = kwargs.pop('hit_count', 0)
        if kwargs:
            raise TypeError('unexpected named args: {}'.format(
                ', '.join(kwargs)))
        super(DexFinishTest, self).__init__()

    def eval(self):
        raise NotImplementedError('DexFinishTest commands cannot be evaled.')

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
