"""
Test jumping to different places.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadJumpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        """Test thread jump handling."""
        self.build(dictionary=self.getBuildFlags())
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Find the line numbers for our breakpoints.
        self.mark1 = line_number('main.cpp', '// 1st marker')
        self.mark2 = line_number('main.cpp', '// 2nd marker')
        self.mark3 = line_number('main.cpp', '// 3rd marker')
        self.mark4 = line_number('main.cpp', '// 4th marker')
        self.mark5 = line_number('other.cpp', '// other marker')

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.mark3, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint 1.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT + " 1",
            substrs=[
                'stopped',
                'main.cpp:{}'.format(
                    self.mark3),
                'stop reason = breakpoint 1'])

        # Try the int path, force it to return 'a'
        self.do_min_test(self.mark3, self.mark1, "i", "4")
        # Try the int path, force it to return 'b'
        self.do_min_test(self.mark3, self.mark2, "i", "5")
        # Try the double path, force it to return 'a'
        self.do_min_test(self.mark4, self.mark1, "j", "7")
        # Try the double path, force it to return 'b'
        self.do_min_test(self.mark4, self.mark2, "j", "8")

        # Try jumping to another function in a different file.
        self.runCmd(
            "thread jump --file other.cpp --line %i --force" %
            self.mark5)
        self.expect("process status",
                    substrs=["at other.cpp:%i" % self.mark5])

        # Try jumping to another function (without forcing)
        self.expect(
            "j main.cpp:%i" %
            self.mark1,
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["error"])

    def do_min_test(self, start, jump, var, value):
        # jump to the start marker
        self.runCmd("j %i" % start)
        self.runCmd("thread step-in")                   # step into the min fn
        # jump to the branch we're interested in
        self.runCmd("j %i" % jump)
        self.runCmd("thread step-out")                  # return out
        self.runCmd("thread step-over")                 # assign to the global
        self.expect("expr %s" % var, substrs=[value])  # check it
