"""Test evaluating expressions which ref. index variable 'i' which just goes
from out of scope to in scope when stopped at the breakpoint."""

from __future__ import print_function


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NonOverlappingIndexVariableCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.cpp'
        self.line_to_break = line_number(
            self.source, '// Set breakpoint here.')

    # rdar://problem/9890530
    def test_eval_index_variable(self):
        """Test expressions of variable 'i' which appears in two for loops."""
        self.build()
        self.exe_name = 'a.out'
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file %s" % exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            self.source,
            self.line_to_break,
            num_expected_locations=1,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.runCmd('frame variable i')
        self.runCmd('expr i')
        self.runCmd('expr ptr[0]->point.x')
        self.runCmd('expr ptr[0]->point.y')
        self.runCmd('expr ptr[i]->point.x')
        self.runCmd('expr ptr[i]->point.y')
