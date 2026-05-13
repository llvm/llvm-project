# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger output to script expected values."""

from typing import Any, Dict, List, Union

from dex.dextIR import ValueIR
from dex.test_script.Nodes import Expect, Value



class DebuggerExpectMatch:
    """Class that represents the match between a particular expected value for an Expect node and the actual debugger
    output corresponding to the watched value for that node."""
    def __init__(self, expect: Expect, expected, actual: ValueIR):
        self.expect = expect
        self.expected = expected
        self.actual = actual
        self.actual_result = self.expect.get_variable_result(self.actual)
        self.match_result = self.expected is not None and str(self.expected) == self.actual_result

def get_expect_match(expect: Expect, expected_values, actual: ValueIR):
    """Given one or more expected values for an Expect node and an actual ValueIR, returns a match for the first
    matching expected values, or for None if there are no matching expected values."""
    if not isinstance(expected_values, list):
        expected_values = [expected_values]
    for expected_value in expected_values:
        expect_match = DebuggerExpectMatch(expect, expected_value, actual)
        if expect_match.match_result:
            return expect_match
    return DebuggerExpectMatch(expect, None, actual)

