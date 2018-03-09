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


class CreateDuringStepTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained")
    @expectedFailureAll(
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="llvm.org/pr15824 thread states not properly maintained, <rdar://problem/28557237>")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24668: Breakpoints not resolved correctly")
    def test_step_inst(self):
        """Test thread creation during step-inst handling."""
        self.build(dictionary=self.getBuildFlags())
        self.create_during_step_base(
            "thread step-inst -m all-threads",
            'stop reason = instruction step')

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained")
    @expectedFailureAll(
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="llvm.org/pr15824 thread states not properly maintained, <rdar://problem/28557237>")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24668: Breakpoints not resolved correctly")
    def test_step_over(self):
        """Test thread creation during step-over handling."""
        self.build(dictionary=self.getBuildFlags())
        self.create_during_step_base(
            "thread step-over -m all-threads",
            'stop reason = step over')

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained")
    @expectedFailureAll(
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="<rdar://problem/28574077>")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24668: Breakpoints not resolved correctly")
    def test_step_in(self):
        """Test thread creation during step-in handling."""
        self.build(dictionary=self.getBuildFlags())
        self.create_during_step_base(
            "thread step-in -m all-threads",
            'stop reason = step in')

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')
        self.continuepoint = line_number('main.cpp', '// Continue from here')

    def create_during_step_base(self, step_cmd, step_stop_reason):
        """Test thread creation while using step-in."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Get the target process
        target = self.dbg.GetSelectedTarget()

        # This should create a breakpoint in the stepping thread.
        self.bkpt = target.BreakpointCreateByLocation("main.cpp", self.breakpoint) 

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        process = target.GetProcess()

        # The stop reason of the thread should be breakpoint.
        stepping_thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, self.bkpt)
        self.assertTrue(stepping_thread.IsValid(), "We stopped at the right breakpoint")

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see only two threads
        self.assertTrue(
            num_threads == 2,
            'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        thread1 = process.GetThreadAtIndex(0)
        thread2 = process.GetThreadAtIndex(1)

        current_line = self.breakpoint
        # Keep stepping until we've reached our designated continue point
        while current_line != self.continuepoint:
            if stepping_thread != process.GetSelectedThread():
                process.SetSelectedThread(stepping_thread)

            self.runCmd(step_cmd)

            frame = stepping_thread.GetFrameAtIndex(0)
            current_line = frame.GetLineEntry().GetLine()

            # Make sure we're still where we thought we were
            self.assertTrue(
                current_line >= self.breakpoint,
                "Stepped to unexpected line, " +
                str(current_line))
            self.assertTrue(
                current_line <= self.continuepoint,
                "Stepped to unexpected line, " +
                str(current_line))

        # Update the number of threads
        num_threads = process.GetNumThreads()

        # Check to see that we increased the number of threads as expected
        self.assertTrue(
            num_threads == 3,
            'Number of expected threads and actual threads do not match after thread exit.')

        stop_reason = stepping_thread.GetStopReason()
        self.assertEqual(stop_reason, lldb.eStopReasonPlanComplete, "Stopped for plan completion")

        # Run to completion
        self.runCmd("process continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(
            process.GetState() == lldb.eStateExited,
            PROCESS_EXITED)
