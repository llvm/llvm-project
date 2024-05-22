# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Commmand sets the path for all following commands to 'declared_file'.
"""

from pathlib import PurePath

from dex.command.CommandBase import CommandBase


class DexDeclareFile(CommandBase):
    def __init__(self, declared_file):
        if not isinstance(declared_file, str):
            raise TypeError("invalid argument type")

        # Use PurePath to create a cannonical platform path.
        # TODO: keep paths as PurePath objects for 'longer'
        self.declared_file = str(PurePath(declared_file))
        super(DexDeclareFile, self).__init__()

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self):
        return self.declared_file
