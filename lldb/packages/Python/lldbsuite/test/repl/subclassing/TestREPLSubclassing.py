# TestREPLSubclassing.py
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
"""This test verifies that the REPL can validly do subclass."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests
from lldbsuite.test import decorators


class REPLSubclassingTestCase (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    @decorators.expectedFailureAll(
        oslist=["macosx"],
        bugnumber="rdar://26768714")
    def doTest(self):
        self.command('class A {init(a: Int) {}}')
        self.command(
            'class B : A {let x: Int; init() { x = 10; super.init(a: x) } }')
        self.command('print(B().x)', patterns=['10'])
        self.command(
            'extension B : CustomStringConvertible { var description:String { return "class B is a subclass of class A"} }')
        self.command(
            'print(B())',
            patterns=['class B is a subclass of class A'])
