# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Command that tells dexter to set a breakpoint which, after hitting,
signals that the debugger should 'continue' until another is hit. Continuing
out of a function being stepped through with DexStepFunction is well defined:
stepping will resume in other functions tracked further down the stacktrace.

NOTE: Only supported for DAP-based debuggers.
"""

from dex.command.CommandBase import CommandBase


class DexContinue(CommandBase):
    def __init__(self, *args, **kwargs):
        # DexContinue(*[expr, *values], **from_line[, **to_line, **hit_count])

        # Optional positional args: expr, values.
        if len(args) == 0:
            self.expression = None
            self.values = []
        elif len(args) == 1:
            raise TypeError("expected 0 or at least 2 positional arguments")
        else:
            self.expression = args[0]
            self.values = [str(arg) for arg in args[1:]]

        # Required keyword arg: from_line.
        try:
            self.from_line = kwargs.pop("from_line")
        except:
            raise TypeError("Missing from_line argument")

        # Optional conditional args: to_line, hit_count.
        self.to_line = kwargs.pop("to_line", None)
        self.hit_count = kwargs.pop("hit_count", None)

        if kwargs:
            raise TypeError("unexpected named args: {}".format(", ".join(kwargs)))
        super(DexContinue, self).__init__()

    def eval(self):
        raise NotImplementedError("DexContinue commands cannot be evaled.")

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
