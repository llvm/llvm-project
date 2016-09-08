# TestSwiftieFormatting.py
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
Test that data formatters honor Swift conventions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftieFormatting(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    def test_swiftie_formatting(self):
        """Test that data formatters honor Swift conventions"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that data formatters honor Swift conventions"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
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

        swcla = self.frame.FindVariable("swcla")
        swcla.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        swcla.SetPreferSyntheticValue(True)

        ns_a = swcla.GetChildMemberWithName("ns_a")
        self.assertTrue(
            ns_a.GetSummary() == '"Hello Swift"',
            "ns_a summary wrong")

        ns_d = swcla.GetChildMemberWithName("ns_d")
        self.assertTrue(ns_d.GetSummary() == '0 bytes', "ns_d summary wrong")

        ns_n = swcla.GetChildMemberWithName("ns_n")
        self.assertTrue(ns_n.GetSummary() == 'Int64(30)', "ns_n summary wrong")

        swcla = self.frame.EvaluateExpression("swcla")
        swcla.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        swcla.SetPreferSyntheticValue(True)

        ns_a = swcla.GetChildMemberWithName("ns_a")
        self.assertTrue(
            ns_a.GetSummary() == '"Hello Swift"',
            "ns_a summary wrong")

        ns_d = swcla.GetChildMemberWithName("ns_d")
        self.assertTrue(ns_d.GetSummary() == '0 bytes', "ns_d summary wrong")

        ns_n = swcla.GetChildMemberWithName("ns_n")
        self.assertTrue(ns_n.GetSummary() == 'Int64(30)', "ns_n summary wrong")

        nsarr = self.frame.FindVariable("nsarr")
        nsarr.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr.SetPreferSyntheticValue(True)

        nsarr0 = nsarr.GetChildAtIndex(0)
        nsarr0.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr0.SetPreferSyntheticValue(True)
        nsarr1 = nsarr.GetChildAtIndex(1)
        nsarr1.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr1.SetPreferSyntheticValue(True)
        nsarr3 = nsarr.GetChildAtIndex(3)
        nsarr3.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        nsarr3.SetPreferSyntheticValue(True)

        self.assertTrue(
            nsarr0.GetSummary() == 'Int64(2)',
            'nsarr[0] summary wrong')
        self.assertTrue(
            nsarr1.GetSummary() == 'Int64(3)',
            'nsarr[1] summary wrong')
        self.assertTrue(
            nsarr3.GetSummary() == 'Int64(5)',
            'nsarr[3] summary wrong')

        self.expect(
            'frame variable -d run nsarr[4] --ptr-depth=1',
            substrs=[
                '"One"',
                '"Two"',
                '"Three"'])
        self.expect(
            'frame variable -d run nsarr[5] --ptr-depth=1',
            substrs=[
                'Int64(1)',
                'Int64(2)',
                'Int64(3)'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
