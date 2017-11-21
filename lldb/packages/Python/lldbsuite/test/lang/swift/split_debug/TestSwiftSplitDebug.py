# TestSplitDebug.py
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
"""
Test that split debug-info works properly
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftSplitDebug(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_split_debug_info(self):
        """Test split debug info"""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_val(self, var_name, expected_val):
        value = self.frame.EvaluateExpression(var_name,
            lldb.eDynamicCanRunTarget)

        self.assertTrue(value.IsValid(), "expr " + var_name + " returned a valid value")
        self.assertEquals(value.GetValue(), expected_val)

    def do_test(self):
        """Test the split debug info"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoint:
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Break here in main', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0,
            lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(self.process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame.IsValid(), "Frame 0 is valid.")

        self.check_val("c.c_x", "12345")
        self.check_val("c.c_y", "6789")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
