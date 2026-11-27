# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Command that tells dexter to set a function breakpoint and step through
the function after hitting it.

NOTE: Only supported for DAP-based debuggers.
"""

from dex.command.CommandBase import CommandBase


class DexStepFunction(CommandBase):
    def __init__(self, *args, **kwargs):
        if len(args) < 1:
            raise TypeError("expected 1 positional argument")
        self.function = str(args[0])
        self.hit_count = kwargs.pop("hit_count", None)
        if kwargs:
            raise TypeError(f"unexpected named args: {', '.join(kwargs)}")
        super(DexStepFunction, self).__init__()

    def eval(self):
        raise NotImplementedError("DexStepFunction commands cannot be evaled.")

    def get_function(self):
        return self.function

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
