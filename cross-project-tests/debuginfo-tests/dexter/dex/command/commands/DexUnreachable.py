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

    def __init__(self, *args, **kwargs):
        if len(args) != 0:
            raise TypeError("DexUnreachable takes no positional arguments")
        if 'on_line' in kwargs:
            on_line = kwargs.pop('on_line')
            self._from_line = on_line
            self._to_line = on_line
        elif 'from_line' in kwargs and 'to_line' in kwargs:
            self._from_line = kwargs.pop('from_line')
            self._to_line = kwargs.pop('to_line')
        elif 'from_line' in kwargs or 'to_line' in kwargs:
            raise TypeError("Must provide both from_line and to_line to DexUnreachable")

        if len(kwargs) > 0:
            raise TypeError("Unexpected kwargs {}".format(kwargs.keys()))
        super(DexUnreachable, self).__init__()
        pass

    @staticmethod
    def get_name():
        return __class__.__name__

    def eval(self, step_info):
        # If we're ever called, at all, then we're evaluating a line that has
        # been marked as unreachable. Which means a failure.
        vir = ValueIR(expression="Unreachable",
                      value="True", type_name=None,
                      error_string=None,
                      could_evaluate=True,
                      is_optimized_away=True,
                      is_irretrievable=False)
        return {'DexUnreachable' : vir}
