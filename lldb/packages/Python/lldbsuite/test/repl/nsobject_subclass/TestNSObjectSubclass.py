# TestNSObjectSubclass.py
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
"""Test that the REPL allows defining subclasses of NSObject"""

import os
import time
import unittest2
import lldb
from lldbsuite.test.lldbrepl import REPLTest, load_tests
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest


class REPLNSObjectSubclassTest (REPLTest):

    mydir = REPLTest.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipUnlessDarwin
    @decorators.no_debug_info_test
    def testREPL(self):
        REPLTest.testREPL(self)

    def doTest(self):
        self.command('import Foundation', timeout=20)
        self.command('''class Foo : NSObject {
          var bar : Int
          var baaz : Int
          init (a: Int, b: Int) {
            bar = a
            baaz = b
          }
          func sum() -> Int {
            return bar + baaz
          }
        }''', timeout=20)
        self.command(
            'Foo(a:2, b:3)',
            patterns=[
                '\\$R0: Foo = {',
                'ObjectiveC\\.NSObject = {',
                'isa = __lldb_expr_[0-9]+\\.Foo',
                'bar = 2',
                'baaz = 3'],
            timeout=20)
        self.command('$R0.sum()', patterns='\\$R1: Int = 5', timeout=20)
