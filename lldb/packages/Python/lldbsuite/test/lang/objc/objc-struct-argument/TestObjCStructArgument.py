"""Test passing structs to Objective-C methods."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCStructArgument(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "test.m"
        self.break_line = line_number(
            self.main_source, '// Set breakpoint here.')

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test passing structs to Objective-C methods."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bpt = target.BreakpointCreateByLocation(
            self.main_source, self.break_line)
        self.assertTrue(bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(process, bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertTrue(len(thread_list) == 1,
                        "More than one thread stopped at our breakpoint.")

        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        self.expect("p [summer sumThings:tts]", substrs=['9'])

        self.expect(
            "po [NSValue valueWithRect:rect]",
            substrs=['NSRect: {{0, 0}, {10, 20}}'])

        # Now make sure we can call a method that returns a struct without
        # crashing.
        cmd_value = frame.EvaluateExpression("[provider getRange]")
        self.assertTrue(cmd_value.IsValid())
