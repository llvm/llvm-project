# TestREPLClasses.py
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
"""Test that we can define and use classes in the REPL"""

import lldbsuite.test.lldbrepl as lldbrepl


class REPLClassesTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('''class foo {
          var bar : Int
          var baaz : Int
          init (a: Int, b: Int) {
            bar = a
            baaz = b
          }
          func sum() -> Int {
            return bar + baaz
          }
        }''')

        self.command(
            'foo(a: 2, b: 3)',
            patterns=[
                r'\$R0',
                r'foo = {',
                r'bar = 2',
                r'baaz = 3'])
