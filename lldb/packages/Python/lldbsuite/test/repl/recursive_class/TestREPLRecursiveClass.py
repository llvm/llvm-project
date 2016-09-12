# TestREPLRecursiveClass.py
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
"""Test that recursive class instances work in the REPL."""

import unittest2

import lldbsuite.test.lldbrepl as lldbrepl
from lldbsuite.test import decorators


class REPLRecursiveClassTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    def doTest(self):
        self.command('''class Foo {
          var aFoo: Foo!
          var x: String = "Hello World"

          init() {
          }
        }''')
        self.command('var a = Foo()', patterns=['aFoo', 'nil'])
        self.command('a.aFoo = a')
        self.command('a', patterns=[
            r'\$R0: Foo = {',
            r'aFoo = 0x[0-9a-fA-F]+ \{\.\.\.\}',
            'x = "Hello World"'])
