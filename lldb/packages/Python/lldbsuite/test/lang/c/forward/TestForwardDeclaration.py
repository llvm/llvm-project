"""Test that forward declaration of a data structure gets resolved correctly."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ForwardDeclarationTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_and_run_command(self):
        """Display *bar_ptr when stopped on a function with forward declaration of struct bar."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_symbol(
            self, "foo", num_expected_locations=1, sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        # Note that the member fields of a = 1 and b = 2 is by design.
        self.expect(
            "frame variable --show-types *bar_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(bar) *bar_ptr = ',
                '(int) a = 1',
                '(int) b = 2'])

        # And so should this.
        self.expect(
            "expression --show-types -- *bar_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(bar)',
                '(int) a = 1',
                '(int) b = 2'])
