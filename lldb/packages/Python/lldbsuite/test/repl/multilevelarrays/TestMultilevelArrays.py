# TestMultilevelArrays.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""Test that nested Arrays work in the REPL."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests


class REPLNestedArrayTestCase (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('[[1,2,3,4],[1,2,3],[1,2],[],[1]]', patterns=[
            '\\$R0: \\[\\[Int]] = 5 values {',
            '\\[0] = 4 values {',
            '\\[0] = 1',
            '\\[1] = 2',
            '\\[2] = 3',
            '\\[3] = 4',
            '\\[1] = 3 values {',
            '\\[0] = 1',
            '\\[1] = 2',
            '\\[2] = 3',
            '\\[2] = 2 values {',
            '\\[0] = 1',
            '\\[1] = 2',
            '\\[3] = 0 values',
            '\\[4] = 1 value {',
            '\\[0] = 1'
        ])
