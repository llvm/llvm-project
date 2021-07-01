# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command for specifying an expected set of types for a particular watch."""


from dex.command.commands.DexExpectWatchBase import DexExpectWatchBase

class DexExpectWatchType(DexExpectWatchBase):
    """Expect the expression `expr` to evaluate be evaluated and have each
    evaluation's type checked against the list of `types`.

    DexExpectWatchType(expr, *types [,**from_line=1][,**to_line=Max]
                        [,**on_line])

    See Commands.md for more info.
    """
    @staticmethod
    def get_name():
        return __class__.__name__

    def _get_expected_field(self, watch):
        return watch.type_name
