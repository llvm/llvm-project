"""
Test calling a function, stopping in the call, continue and gather the result on stop.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandCallStopContinueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Please test these expressions while stopped at this line:')
        self.func_line = line_number('main.cpp', '{5, "five"}')

    @expectedFlakeyDarwin("llvm.org/pr20274")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24489: Name lookup not working correctly on Windows")
    def test(self):
        """Test gathering result from interrupted function call."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        # Some versions of GCC encode two locations for the 'return' statement
        # in main.cpp
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.func_line,
            num_expected_locations=-1,
            loc_exact=True)

        self.expect("expr -i false -- returnsFive()", error=True,
                    substrs=['Execution was interrupted, reason: breakpoint'])

        self.runCmd("continue", "Continue completed")
        self.expect(
            "thread list",
            substrs=[
                'stop reason = User Expression thread plan',
                r'Completed expression: (Five) $0 = (number = 5, name = "five")'])
