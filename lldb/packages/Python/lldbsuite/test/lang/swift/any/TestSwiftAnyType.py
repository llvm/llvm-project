# TestSwiftAnyType.py
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
Test the Any type
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftAnyType(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.expectedFailureAll(bugnumber='rdar://27527768', oslist=['linux'])
    def test_any_type(self):
        """Test the Any type"""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)

    def do_test(self):
        """Test the Any type"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex('Set breakpoint here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var_c = self.frame.FindVariable("c")
        var_c_x = var_c.GetChildMemberWithName("x")
        var_q = self.frame.FindVariable("q")
        lldbutil.check_variable(self,var_c_x,True,value="12")
        lldbutil.check_variable(self,var_q,True,value="12")

        self.expect("expression -d run -- q", substrs=['12'])
        self.expect("frame variable -d run -- q", substrs=['12'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
