# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A Command that specifies the command line with which to run the test.
"""

from dex.command.CommandBase import CommandBase


class DexCommandLine(CommandBase):
    def __init__(self, the_cmdline):
        if type(the_cmdline) is not list:
            raise TypeError("Expected list, got {}".format(type(the_cmdline)))
        for x in the_cmdline:
            if type(x) is not str:
                raise TypeError(
                    'Command line element "{}" has type {}'.format(x, type(x))
                )
        self.the_cmdline = the_cmdline
        super(DexCommandLine, self).__init__()

    def eval(self):
        raise NotImplementedError("DexCommandLine commands cannot be evaled.")

    @staticmethod
    def get_name():
        return __class__.__name__

    @staticmethod
    def get_subcommands() -> dict:
        return None
