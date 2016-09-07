# TestSwiftNestedArray.py
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
Test Arrays of Arrays in Swift
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


def check_for_idx(child, idx):
    return child.GetValue() == str(idx + 1)


def check_for_C(child, idx):
    if child.GetTypeName() == "a.C":
        if child.GetNumChildren() == 1:
            if child.GetChildAtIndex(0).GetName() == "m_counter":
                return True
    return False


class TestSwiftNestedArray(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_nested_array(self):
        """Test Arrays of Arrays in Swift"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test Arrays of Arrays in Swift"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            '// break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var_aInt = self.frame.FindVariable("aInt")
        var_aC = self.frame.FindVariable("aC")
        lldbutil.check_variable(self, var_aInt, False, num_children=6)
        lldbutil.check_variable(self, var_aC, False, num_children=5)

        for i in range(0, 6):
            var_aIntChild = var_aInt.GetChildAtIndex(i)
            lldbutil.check_children(self, var_aIntChild, check_for_idx)

        for i in range(0, 5):
            var_aCChild = var_aC.GetChildAtIndex(i)
            lldbutil.check_children(self, var_aCChild, check_for_C)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
