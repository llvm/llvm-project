# TestREPLBreakpoints.py
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

import os
import time
import unittest2
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbrepl as lldbrepl


class REPLBreakpointsTestCase (lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.no_debug_info_test
    @decorators.expectedFailureAll(
        oslist=[
            "macosx",
            "linux"],
        bugnumber="rdar://23091701")
    def testREPL(self):
        REPLTest.testREPL(self)

    def doTest(self):
        self.command('''func foo() {
    print("hello")
}''', prompt_sync=False, patterns=['4>'])

        # Set a breakpoint
        function_pattern = '''foo \(\) -> \(\)'''
        source_pattern = 'at repl.swift:2'
        self.command(
            ':b 2',
            prompt_sync=False,
            patterns=[
                'Breakpoint 1',
                function_pattern,
                source_pattern,
                'address = 0x',
                '4>'])
        self.command(
            'foo()',
            prompt_sync=False,
            patterns=[
                'Execution stopped at breakpoint',
                'Process [0-9]+ stopped',
                'thread #1: tid = 0x',
                'foo\(\) -> \(\)',
                source_pattern,
                'stop reason = breakpoint 1.1',
                '-> 2',
                '''print\("hello"\)'''])
