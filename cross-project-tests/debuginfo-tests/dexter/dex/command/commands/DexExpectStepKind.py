# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command for specifying an expected number of steps of a particular kind."""

from dex.command.CommandBase import CommandBase
from dex.dextIR.StepIR import StepKind


class DexExpectStepKind(CommandBase):
    """Expect to see a particular step `kind` a number of `times` while stepping
    through the program.

    DexExpectStepKind(kind, times)

    See Commands.md for more info.
    """

    def __init__(self, *args):
        if len(args) != 2:
            raise TypeError('expected two args')

        try:
            step_kind = StepKind[args[0]]
        except KeyError:
            raise TypeError('expected arg 0 to be one of {}'.format(
                [kind for kind, _ in StepKind.__members__.items()]))

        self.name = step_kind
        self.count = args[1]

        super(DexExpectStepKind, self).__init__()

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self):
        # DexExpectStepKind eval() implementation is mixed into
        # Heuristic.__init__()
        # [TODO] Fix this ^.
        pass
