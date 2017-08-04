# TestREPLSubclassing.py
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
"""This test verifies that the REPL can validly do subclass."""

import os
import time
import unittest2
import lldb
import lldbsuite.test.lldbrepl as lldbrepl
from lldbsuite.test import decorators


class REPLSubclassingTestCase (lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('class A {init(a: Int) {}}')
        self.command(
            'class B : A {let x: Int; init() { x = 5 + 5; super.init(a: x) } }')
        self.command('print(B().x)', patterns=['10'])
        self.command(
            'extension B : CustomStringConvertible { public var description:String { return "class B\(x) is a subclass of class A"} }')
        self.command(
            'print(B())',
            patterns=['class B(10) is a subclass of class A'])
