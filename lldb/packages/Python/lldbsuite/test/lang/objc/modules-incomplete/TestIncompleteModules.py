"""Test that DWARF types are trusted over module types"""

from __future__ import print_function


import unittest2
import platform
from distutils.version import StrictVersion

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class IncompleteModulesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    @skipUnlessDarwin
    @unittest2.expectedFailure("rdar://20416388")
    @unittest2.skipIf(platform.system() != "Darwin" or StrictVersion(
        '12.0.0') > platform.release(), "Only supported on Darwin 12.0.0+")
    @skipIfDarwin  # llvm.org/pr26267
    def test_expr(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.runCmd(
            "settings set target.clang-module-search-paths \"" +
            os.getcwd() +
            "\"")

        self.expect("expr @import myModule; 3", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["int", "3"])

        self.expect(
            "expr [myObject privateMethod]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "int",
                "5"])

        self.expect("expr MIN(2,3)", "#defined macro was found",
                    substrs=["int", "2"])

        self.expect("expr MAX(2,3)", "#undefd macro was correcltly not found",
                    error=True)
