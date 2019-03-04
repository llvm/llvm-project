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


class MultipleBreakpointTestCase(TestBase):

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
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="llvm.org/pr15824 thread states not properly maintained and <rdar://problem/28557237>")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    @skipIfWindows # This is flakey on Windows: llvm.org/pr24668, llvm.org/pr38373
    @expectedFailureNetBSD
    def test(self):
        """Test simultaneous breakpoints in multiple threads."""
        self.build(dictionary=self.getBuildFlags())
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        # The breakpoint may be hit in either thread 2 or thread 3.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see all three threads
        self.assertTrue(
            num_threads >= 3,
            'Number of expected threads and actual threads do not match.')

        # Get the thread objects
        thread1 = process.GetThreadAtIndex(0)
        thread2 = process.GetThreadAtIndex(1)
        thread3 = process.GetThreadAtIndex(2)

        # Make sure both threads are stopped
        self.assertTrue(
            thread1.IsStopped(),
            "Primary thread didn't stop during breakpoint")
        self.assertTrue(
            thread2.IsStopped(),
            "Secondary thread didn't stop during breakpoint")
        self.assertTrue(
            thread3.IsStopped(),
            "Tertiary thread didn't stop during breakpoint")

        # Delete the first breakpoint then continue
        self.runCmd("breakpoint delete 1")

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(
            process.GetState() == lldb.eStateExited,
            PROCESS_EXITED)
