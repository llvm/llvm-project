# TestREPLQuickLookObject.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""Test that QuickLookObject works correctly in the REPL"""

import os
import time
import unittest2
import lldb
import lldbsuite.test.lldbrepl as lldbrepl
import lldbsuite.test.lldbtest as lldbtest


class REPLQuickLookTestCase (lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command(
            'PlaygroundQuickLook(reflecting: true)',
            patterns=['bool = true'])
        self.command(
            'PlaygroundQuickLook(reflecting: 1.25)',
            patterns=['double = 1.25'])
        self.command(
            'PlaygroundQuickLook(reflecting: Float(1.25))',
            patterns=['float = 1.25'])
        self.command(
            'PlaygroundQuickLook(reflecting: "Hello")',
            patterns=['text = \"Hello\"'])
