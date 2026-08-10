# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Commmand sets the path for all following commands to 'declared_file'.
"""

import os
from pathlib import PurePath

from dex.command.CommandBase import CommandBase, StepExpectInfo


class DexDeclareAddress(CommandBase):
    def __init__(self, addr_name, expression, **kwargs):
        if not isinstance(addr_name, str):
            raise TypeError("invalid argument type")

        self.addr_name = addr_name
        self.expression = expression
        self.on_line = kwargs.pop("on_line")
        self.hit_count = kwargs.pop("hit_count", 0)

        self.address_resolutions = None

        super(DexDeclareAddress, self).__init__()

    @staticmethod
    def get_name():
        return __class__.__name__

    def get_watches(self):
        return [
            StepExpectInfo(
                self.expression, self.path, 0, range(self.on_line, self.on_line + 1)
            )
        ]

    def get_address_name(self):
        return self.addr_name

    def eval(self, step_collection):
        self.address_resolutions[self.get_address_name()] = None
        for step in step_collection.steps:
            loc = step.current_location

            if (
                loc.path
                and self.path
                and PurePath(loc.path) == PurePath(self.path)
                and loc.lineno == self.on_line
            ):
                if self.hit_count > 0:
                    self.hit_count -= 1
                    continue
                try:
                    watch = step.program_state.frames[0].watches[self.expression]
                except KeyError:
                    continue
                try:
                    hex_val = int(watch.value, 16)
                except ValueError:
                    hex_val = None
                self.address_resolutions[self.get_address_name()] = hex_val
                break
