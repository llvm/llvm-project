"""
Test number of threads.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class ThreadExitTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.break_1 = line_number('main.cpp', '// Set first breakpoint here')
        self.break_2 = line_number('main.cpp', '// Set second breakpoint here')
        self.break_3 = line_number('main.cpp', '// Set third breakpoint here')
        self.break_4 = line_number('main.cpp', '// Set fourth breakpoint here')

    @expectedFailureWindows("llvm.org/pr24681")
    def test(self):
        """Test thread exit handling."""
        self.build(dictionary=self.getBuildFlags())
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 location.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.break_1, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.break_2, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.break_3, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.break_4, num_expected_locations=1)

        # The breakpoint list should show 1 locations.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.break_1,
                       "2: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.break_2,
                       "3: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.break_3,
                       "4: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.break_4])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint 1.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT + " 1",
            substrs = ['stopped',
                       '* thread #1',
                       'stop reason = breakpoint 1',
                       'thread #2'])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        self.assertTrue(num_threads == 2, 'Number of expected threads and actual threads do not match at breakpoint 1.')

        # Run to the second breakpoint
        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint 1.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT + " 2",
            substrs = ['stopped',
                       'thread #1',
                       'thread #2',
                       'stop reason = breakpoint 2',
                       'thread #3'])

        # Update the number of threads
        num_threads = process.GetNumThreads()

        self.assertTrue(num_threads == 3, 'Number of expected threads and actual threads do not match at breakpoint 2.')

        # Run to the third breakpoint
        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint 3.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT + " 3",
            substrs = ['stopped',
                       'thread #1',
                       'stop reason = breakpoint 3',
                       'thread #3',
                       ])

        # Update the number of threads
        num_threads = process.GetNumThreads()

        self.assertTrue(num_threads == 2, 'Number of expected threads and actual threads do not match at breakpoint 3.')

        # Run to the fourth breakpoint
        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint 4.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT + " 4",
            substrs = ['stopped',
                       'thread #1',
                       'stop reason = breakpoint 4'])

        # Update the number of threads
        num_threads = process.GetNumThreads()

        self.assertTrue(num_threads == 1, 'Number of expected threads and actual threads do not match at breakpoint 4.')

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)
