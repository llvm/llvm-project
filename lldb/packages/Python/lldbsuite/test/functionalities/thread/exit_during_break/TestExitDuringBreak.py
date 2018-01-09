"""
Test number of threads.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExitDuringBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    def test(self):
        """Test thread exit during breakpoint handling."""
        self.build(dictionary=self.getBuildFlags())
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # The exit probably occurred during breakpoint handling, but it isn't
        # guaranteed.  The main thing we're testing here is that the debugger
        # handles this cleanly is some way.

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see at least five threads
        self.assertTrue(
            num_threads >= 5,
            'Number of expected threads and actual threads do not match.')

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(
            process.GetState() == lldb.eStateExited,
            PROCESS_EXITED)
