# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from dex.command.CommandBase import CommandBase
from dex.dextIR import ValueIR


class DexUnreachable(CommandBase):
    """Expect the source line this is found on will never be stepped on to.

    DexUnreachable()

    See Commands.md for more info.
    """

    def __init(self):
        super(DexUnreachable, self).__init__()
        pass

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self, debugger):
        # If we're ever called, at all, then we're evaluating a line that has
        # been marked as unreachable. Which means a failure.
        vir = ValueIR(expression="Unreachable",
                      value="True", type_name=None,
                      error_string=None,
                      could_evaluate=True,
                      is_optimized_away=True,
                      is_irretrievable=False)
        return {'DexUnreachable' : vir}
