# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class StepValueInfo(object):
    def __init__(self, step_index, watch_info, expected_value):
        self.step_index = step_index
        self.watch_info = watch_info
        self.expected_value = expected_value

    def __str__(self):
        return '{}:{}: expected value:{}'.format(self.step_index, self.watch_info, self.expected_value)

    def __eq__(self, other):
        return (self.watch_info.expression == other.watch_info.expression
                and self.expected_value == other.expected_value)

    def __hash__(self):
        return hash(self.watch_info.expression, self.expected_value)
