"""
Test breakpoint hit count features.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointHitCountTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_breakpoint_location_hit_count(self):
        """Use Python APIs to check breakpoint hit count."""
        self.build()
        self.do_test_breakpoint_location_hit_count()

    def test_breakpoint_one_shot(self):
        """Check that one-shot breakpoints trigger only once."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.runCmd("tb a")
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.GetFunctionName() == "a(int)" or frame0.GetFunctionName() == "int a(int)");

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.a_int_body_line_no = line_number(
            'main.cpp', '// Breakpoint Location 1')
        self.a_float_body_line_no = line_number(
            'main.cpp', '// Breakpoint Location 2')

    def do_test_breakpoint_location_hit_count(self):
        """Use Python APIs to check breakpoint hit count."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint in main.cpp by name 'a',
        # there should be two locations.
        breakpoint = target.BreakpointCreateByName('a', 'a.out')
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 2,
                        VALID_BREAKPOINT)

        # Verify all breakpoint locations are enabled.
        location1 = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location1 and
                        location1.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        location2 = breakpoint.GetLocationAtIndex(1)
        self.assertTrue(location2 and
                        location2.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Verify 1st breakpoint location is hit.
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame0 = thread.GetFrameAtIndex(0)
        location1 = breakpoint.FindLocationByAddress(frame0.GetPC())
        self.assertTrue(
            frame0.GetLineEntry().GetLine() == self.a_int_body_line_no,
            "Stopped in int a(int)")
        self.assertTrue(location1)
        self.assertEqual(location1.GetHitCount(), 1)
        self.assertEqual(breakpoint.GetHitCount(), 1)

        process.Continue()

        # Verify 2nd breakpoint location is hit.
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame0 = thread.GetFrameAtIndex(0)
        location2 = breakpoint.FindLocationByAddress(frame0.GetPC())
        self.assertTrue(
            frame0.GetLineEntry().GetLine() == self.a_float_body_line_no,
            "Stopped in float a(float)")
        self.assertTrue(location2)
        self.assertEqual(location2.GetHitCount(), 1)
        self.assertEqual(location1.GetHitCount(), 1)
        self.assertEqual(breakpoint.GetHitCount(), 2)

        process.Continue()

        # Verify 2nd breakpoint location is hit again.
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        self.assertEqual(location2.GetHitCount(), 2)
        self.assertEqual(location1.GetHitCount(), 1)
        self.assertEqual(breakpoint.GetHitCount(), 3)

        process.Continue()
