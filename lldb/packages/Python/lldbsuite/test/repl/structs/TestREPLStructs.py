# TestREPLStructs.py
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
"""Test that we can define and use structs in the REPL"""

import lldbsuite.test.lldbrepl as lldbrepl
import lldbsuite.test.decorators as decorators


class REPLStructsTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    @decorators.expectedFailureAll("rdar://problem/23545959")
    def doTest(self):
        self.command('''struct foo {
          var bar : Int
          var baaz : Int
        }''')
        self.command(
            'foo(bar: 2, baaz: 3)',
            patterns=[
                r'\$R0: foo = {',
                r'bar = 2',
                r'baaz = 3'])
