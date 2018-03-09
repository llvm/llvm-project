# TestSwiftGenericTupleLabels.py
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
Test that LLDB can reconstruct tuple labels from metadata
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericTupleLabels(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_generic_tuple_labels(self):
        """Test that LLDB can reconstruct tuple labels from metadata"""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that LLDB can reconstruct tuple labels from metadata"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0,
            lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        the_tuple = self.frame.FindVariable('x')
        the_tuple.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        the_tuple.SetPreferSyntheticValue(True)

        self.assertTrue(the_tuple.GetChildAtIndex(
            0).GetName() == 'x', '.0 == x')
        self.assertTrue(the_tuple.GetChildAtIndex(
            1).GetName() == '1', '.1 == 1')
        self.assertTrue(the_tuple.GetChildAtIndex(
            2).GetName() == 'z', '.2 == z')
        self.assertTrue(the_tuple.GetChildAtIndex(
            3).GetName() == '3', '.3 == 3')
        self.assertTrue(the_tuple.GetChildAtIndex(
            4).GetName() == 'q', '.4 == q')
        self.assertTrue(the_tuple.GetChildAtIndex(
            5).GetName() == 'w', '.5 == q')

        self.expect('frame variable -d run -- x.w', substrs=['72'])
        self.expect('expression -d run -- x.z', substrs=['36'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
