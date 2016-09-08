"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class VarInAggregateMisuseTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd(
            "type summary add --summary-string \"SUMMARY SUCCESS ${var}\" Summarize")

        self.expect('frame variable mine_ptr',
                    substrs=['SUMMARY SUCCESS summarize_ptr_t @ '])

        self.expect('frame variable *mine_ptr',
                    substrs=['SUMMARY SUCCESS summarize_t @'])

        self.runCmd(
            "type summary add --summary-string \"SUMMARY SUCCESS ${var.first}\" Summarize")

        self.expect('frame variable mine_ptr',
                    substrs=['SUMMARY SUCCESS 10'])

        self.expect('frame variable *mine_ptr',
                    substrs=['SUMMARY SUCCESS 10'])

        self.runCmd("type summary add --summary-string \"${var}\" Summarize")
        self.runCmd(
            "type summary add --summary-string \"${var}\" -e TwoSummarizes")

        self.expect('frame variable',
                    substrs=['(TwoSummarizes) twos = TwoSummarizes @ ',
                             'first = summarize_t @ ',
                             'second = summarize_t @ '])

        self.runCmd(
            "type summary add --summary-string \"SUMMARY SUCCESS ${var.first}\" Summarize")
        self.expect('frame variable',
                    substrs=['(TwoSummarizes) twos = TwoSummarizes @ ',
                             'first = SUMMARY SUCCESS 1',
                             'second = SUMMARY SUCCESS 3'])
