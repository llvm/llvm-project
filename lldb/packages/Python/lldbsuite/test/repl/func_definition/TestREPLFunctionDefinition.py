# TestREPLFunctionDefinition.py
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
"""Test that we can launch define functions"""

import os
import time
import unittest2
import lldb
import lldbsuite.test.lldbrepl as lldbrepl


class REPLFuncDefinitionTestCase (lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command(
            'func greet(_ name: String) -> String {\nlet greeting = "Hello, " + name + "!"\nreturn greeting\n}')
        self.command('greet("Enrico")', patterns=[
                     '\\$R0: String = "Hello, Enrico!"'])
