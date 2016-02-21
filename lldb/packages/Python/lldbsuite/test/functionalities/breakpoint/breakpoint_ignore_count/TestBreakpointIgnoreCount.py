"""
Test breakpoint ignore count features.
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class BreakpointIgnoreCountTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        self.build()
        self.breakpoint_ignore_count()

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Use Python APIs to set breakpoint ignore count."""
        self.build()
        self.breakpoint_ignore_count_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number of function "c" here.')
        self.line2 = line_number('main.c', '// b(2) -> c(2) Find the call site of b(2).')
        self.line3 = line_number('main.c', '// a(3) -> c(3) Find the call site of c(3).')
        self.line4 = line_number('main.c', '// a(3) -> c(3) Find the call site of a(3).')
        self.line5 = line_number('main.c', '// Find the call site of c in main.')

    def breakpoint_ignore_count(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a breakpoint in main.c at line1.
        lldbutil.run_break_set_by_file_and_line (self, 'main.c', self.line1, extra_options='-i 1', num_expected_locations=1, loc_exact=True)

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* stopped'])

        # Also check the hit count, which should be 2, due to ignore count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_THRICE,
            substrs = ["resolved = 1",
                       "hit count = 2"])

        # The frame #0 should correspond to main.c:37, the executable statement
        # in function name 'c'.  And frame #2 should point to main.c:45.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
            #substrs = ["stop reason = breakpoint"],
            patterns = ["frame #0.*main.c:%d" % self.line1,
                        "frame #2.*main.c:%d" % self.line2])

        # continue -i 1 is the same as setting the ignore count to 1 again, try that:
        # Now run the program.
        self.runCmd("process continue -i 1", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
            patterns = ['Process .* stopped'])

        # Also check the hit count, which should be 2, due to ignore count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_THRICE,
            substrs = ["resolved = 1",
                       "hit count = 4"])

        # The frame #0 should correspond to main.c:37, the executable statement
        # in function name 'c'.  And frame #2 should point to main.c:45.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
            #substrs = ["stop reason = breakpoint"],
            patterns = ["frame #0.*main.c:%d" % self.line1,
                        "frame #1.*main.c:%d" % self.line5])

        

    def breakpoint_ignore_count_python(self):
        """Use Python APIs to set breakpoint ignore count."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Set the ignore count on the breakpoint location.
        location.SetIgnoreCount(2)
        self.assertTrue(location.GetIgnoreCount() == 2,
                        "SetIgnoreCount() works correctly")

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame#0 should be on main.c:37, frame#1 should be on main.c:25, and
        # frame#2 should be on main.c:48.
        #lldbutil.print_stacktraces(process)
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line1 and
                        frame1.GetLineEntry().GetLine() == self.line3 and
                        frame2.GetLineEntry().GetLine() == self.line4,
                        STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT)

        # The hit count for the breakpoint should be 3.
        self.assertTrue(breakpoint.GetHitCount() == 3)

        process.Continue()
