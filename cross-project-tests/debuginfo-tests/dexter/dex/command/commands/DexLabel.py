# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command used to give a line in a test a named psuedonym. Every DexLabel has
   a line number and Label string component.
"""

from dex.command.CommandBase import CommandBase


class DexLabel(CommandBase):
    def __init__(self, label, **kwargs):
        if not isinstance(label, str):
            raise TypeError("invalid argument type")

        try:
            self.on_line = kwargs.pop("on_line")
        except KeyError:
            # We cannot use self.lineno because it hasn't been set yet.
            pass
        if kwargs:
            raise TypeError(f'unexpected named args: {", ".join(kwargs)}')

        self._label = label
        super(DexLabel, self).__init__()

    def get_line(self):
        return getattr(self, "on_line", self.lineno)

    def get_as_pair(self):
        return (self._label, self.get_line())

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self):
        return self._label
