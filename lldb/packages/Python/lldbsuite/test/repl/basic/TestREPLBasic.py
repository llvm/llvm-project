# TestREPLBasic.py
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
"""Test that we can launch the REPL and define a variable."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests


class REPLBasicTestCase (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('var a = 1', patterns=['a: Int = 1'])
        self.command(
            'let b = "Hello World"',
            patterns=['b: String = "Hello World"'])
