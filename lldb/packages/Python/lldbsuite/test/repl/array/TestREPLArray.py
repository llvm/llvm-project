# TestREPLArray.py
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
"""Test that Arrays work in the REPL."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests


class REPLArrayTestCase (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('[2,3,4]', patterns='3 values')
        self.command('$R0[0]', patterns='\\$R1: Int = 2')
        self.command('$R0[1]', patterns='\\$R2: Int = 3')
        self.command('$R0[2]', patterns='\\$R3: Int = 4')
