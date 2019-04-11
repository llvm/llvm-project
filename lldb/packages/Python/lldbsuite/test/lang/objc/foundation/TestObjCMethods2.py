"""
Test more expression command sequences with objective-c.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationTestCase2(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_expr_commands(self):
        """More expression commands for objective-c."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lines = []
        lines.append(
            line_number(
                'main.m',
                '// Break here for selector: tests'))
        lines.append(
            line_number(
                'main.m',
                '// Break here for NSArray tests'))
        lines.append(
            line_number(
                'main.m',
                '// Break here for NSString tests'))
        lines.append(
            line_number(
                'main.m',
                '// Break here for description test'))
        lines.append(
            line_number(
                'main.m',
                '// Set break point at this line'))

        # Create a bunch of breakpoints.
        for line in lines:
            lldbutil.run_break_set_by_file_and_line(
                self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_Selector:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(sel)",
                    substrs=["(char *)",
                             "length"])

        self.runCmd("process continue")

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_MyString:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(_cmd)",
                    substrs=["(char *)",
                             "description"])

        self.runCmd("process continue")
