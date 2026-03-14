# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command to instruct the debugger to inspect the value of some set of
expressions on the current source line.
"""

from dex.command.CommandBase import CommandBase


class DexWatch(CommandBase):
    """[Deprecated] Evaluate each given `expression` when the debugger steps onto the
    line this command is found on

    DexWatch(*expressions)

    See Commands.md for more info.
    """

    def __init__(self, *args):
        if not args:
            raise TypeError("expected some arguments")

        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("invalid argument type")

        self._args = args
        super(DexWatch, self).__init__()

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self, debugger):
        return {arg: debugger.evaluate_expression(arg) for arg in self._args}
