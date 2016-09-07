"""
Test PHI nodes work in the IR interpreter.
"""

import os
import os.path

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class IRInterpreterPHINodesTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test_phi_node_support(self):
        """Test support for PHI nodes in the IR interpreter."""

        self.build()
        exe = os.path.join(os.getcwd(), 'a.out')
        self.runCmd('file ' + exe, CURRENT_EXECUTABLE_SET)

        # Break on the first assignment to i
        line = line_number('main.cpp', 'i = 5')
        lldbutil.run_break_set_by_file_and_line(
            self, 'main.cpp', line, num_expected_locations=1, loc_exact=True)

        self.runCmd('run', RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint
        self.expect('thread list', STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.runCmd('s')

        # The logical 'or' causes a PHI node to be generated. Execute without JIT
        # to test that the interpreter can handle this
        self.expect('expr -j 0 -- i == 3 || i == 5', substrs=['true'])

        self.runCmd('s')
        self.expect('expr -j 0 -- i == 3 || i == 5', substrs=['false'])
        self.runCmd('s')
        self.expect('expr -j 0 -- i == 3 || i == 5', substrs=['true'])
