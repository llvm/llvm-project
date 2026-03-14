# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class ValueIR:
    """Data class to store the result of an expression evaluation."""

    def __init__(
        self,
        expression: str,
        value: str,
        type_name: str,
        could_evaluate: bool,
        error_string: str = None,
        is_optimized_away: bool = False,
        is_irretrievable: bool = False,
    ):
        self.expression = expression
        self.value = value
        self.type_name = type_name
        self.could_evaluate = could_evaluate
        self.error_string = error_string
        self.is_optimized_away = is_optimized_away
        self.is_irretrievable = is_irretrievable

    def __str__(self):
        prefix = '"{}": '.format(self.expression)
        if self.error_string is not None:
            return prefix + self.error_string
        if self.value is not None:
            return prefix + "({}) {}".format(self.type_name, self.value)
        return (
            prefix
            + "could_evaluate: {}; irretrievable: {}; optimized_away: {};".format(
                self.could_evaluate, self.is_irretrievable, self.is_optimized_away
            )
        )
