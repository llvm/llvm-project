"""
Test number of threads.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExitDuringStepTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD # llvm.org/pr21411: test is hanging
    @expectedFlakeyAndroid("llvm.org/pr26206")
    def test(self):
        """Test thread exit during step handling."""
        self.build(dictionary=self.getBuildFlags())
        self.exit_during_step_base("thread step-inst -m all-threads", 'stop reason = instruction step')

    @skipIfFreeBSD # llvm.org/pr21411: test is hanging
    @expectedFlakeyAndroid("llvm.org/pr26206")
    def test_step_over(self):
        """Test thread exit during step-over handling."""
        self.build(dictionary=self.getBuildFlags())
        self.exit_during_step_base("thread step-over -m all-threads", 'stop reason = step over')

    @skipIfFreeBSD # llvm.org/pr21411: test is hanging
    @expectedFlakeyAndroid("llvm.org/pr26206")
    def test_step_in(self):
        """Test thread exit during step-in handling."""
        self.build(dictionary=self.getBuildFlags())
        self.exit_during_step_base("thread step-in -m all-threads", 'stop reason = step in')

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')
        self.continuepoint = line_number('main.cpp', '// Continue from here')

    def exit_during_step_base(self, step_cmd, step_stop_reason):
        """Test thread exit during step handling."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        self.bp_num = lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # The breakpoint list should show 1 location.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.breakpoint])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        num_threads = process.GetNumThreads()
        # Make sure we see all three threads
        self.assertGreaterEqual(num_threads, 3, 'Number of expected threads and actual threads do not match.')

        stepping_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(process, self.bp_num)
        self.assertIsNotNone(stepping_thread, "Could not find a thread stopped at the breakpoint")

        current_line = self.breakpoint
        stepping_frame = stepping_thread.GetFrameAtIndex(0)
        self.assertEqual(current_line, stepping_frame.GetLineEntry().GetLine(), "Starting line for stepping doesn't match breakpoint line.")

        # Keep stepping until we've reached our designated continue point
        while current_line != self.continuepoint:
            # Since we're using the command interpreter to issue the thread command
            # (on the selected thread) we need to ensure the selected thread is the
            # stepping thread.
            if stepping_thread != process.GetSelectedThread():
                process.SetSelectedThread(stepping_thread)

            self.runCmd(step_cmd)

            frame = stepping_thread.GetFrameAtIndex(0)

            current_line = frame.GetLineEntry().GetLine()

            self.assertGreaterEqual(current_line, self.breakpoint, "Stepped to unexpected line, " + str(current_line))
            self.assertLessEqual(current_line, self.continuepoint, "Stepped to unexpected line, " + str(current_line))

        self.runCmd("thread list")

        # Update the number of threads
        new_num_threads = process.GetNumThreads()

        # Check to see that we reduced the number of threads as expected
        self.assertEqual(new_num_threads, num_threads-1, 'Number of threads did not reduce by 1 after thread exit.')

        self.expect("thread list", 'Process state is stopped due to step',
                substrs = ['stopped',
                           step_stop_reason])

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
