# TestSwiftErrorBreakpoint.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Tests catching thrown errors in using the language breakpoint
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import swiftTest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftErrorBreakpoint(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @swiftTest
    def test_swift_error_no_pattern(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        self.do_test(lldb.SBStringList(), True)

    @swiftTest
    def test_swift_error_matching_base_pattern(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        pattern = lldb.SBStringList()
        pattern.AppendString("exception-typename")
        pattern.AppendString("EnumError")
        self.do_test(pattern, True)

    @swiftTest
    def test_swift_error_matching_full_pattern(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        pattern = lldb.SBStringList()
        pattern.AppendString("exception-typename")
        pattern.AppendString("a.EnumError")
        self.do_test(pattern, True)

    @swiftTest
    def test_swift_error_bogus_pattern(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        pattern = lldb.SBStringList()
        pattern.AppendString("exception-typename")
        pattern.AppendString("NoSuchErrorHere")
        self.do_test(pattern, False)

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self, patterns, should_stop):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""

        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        swift_error_bkpt = target.BreakpointCreateForException(
            lldb.eLanguageTypeSwift, False, True, patterns)
        # Note, I'm not checking locations here because we never know them
        # before launch.

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        if should_stop:
            self.assertTrue(process, PROCESS_IS_VALID)
            breakpoint_threads = lldbutil.get_threads_stopped_at_breakpoint(
                process, swift_error_bkpt)
            self.assertTrue(
                len(breakpoint_threads) == 1,
                "We didn't stop at the error breakpoint")
        else:
            exit_state = process.GetState()
            self.assertTrue(
                exit_state == lldb.eStateExited,
                "We stopped at the error breakpoint when we shouldn't have.")

        target.BreakpointDelete(swift_error_bkpt.GetID())

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
