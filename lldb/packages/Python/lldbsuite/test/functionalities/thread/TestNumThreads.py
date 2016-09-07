"""
Test number of threads.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NumberOfThreadsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test(self):
        """Test number of threads."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 location.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1)

        # The breakpoint list should show 3 locations.
        self.expect(
            "breakpoint list -f",
            "Breakpoint location shown correctly",
            substrs=[
                "1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" %
                self.line])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint 1."])

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Using std::thread may involve extra threads, so we assert that there are
        # at least 4 rather than exactly 4.
        self.assertTrue(
            num_threads >= 4,
            'Number of expected threads and actual threads do not match.')
