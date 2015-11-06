# TestREPLPOReplType.py
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
"""Test that types defined in the REPL can be po'ed."""

import os, time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests

class REPLPOTestCase (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('struct S {}')
        self.command(':po S()', patterns=['S'])
        self.command('extension S : CustomDebugStringConvertible { public var debugDescription: String { get { return "ABC" } } }')
        self.command(':po S()', patterns='ABC')

