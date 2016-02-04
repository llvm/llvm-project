"""
Test lldb target stop-hook command.
"""

from __future__ import print_function



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class StopHookCmdTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers inside main.cpp.
        self.begl = line_number('main.cpp', '// Set breakpoint here to test target stop-hook.')
        self.endl = line_number('main.cpp', '// End of the line range for which stop-hook is to be run.')
        self.line = line_number('main.cpp', '// Another breakpoint which is outside of the stop-hook range.')

    @no_debug_info_test
    def test_not_crashing_if_no_target(self):
        """target stop-hook list should not crash if no target has been set."""
        self.runCmd("target stop-hook list", check=False)

    def test(self):
        """Test a sequence of target stop-hook commands."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.begl, num_expected_locations=1, loc_exact=True)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("target stop-hook add -f main.cpp -l %d -e %d -o 'expr ptr'" % (self.begl, self.endl))

        self.expect('target stop-hook list', 'Stop Hook added successfully',
            substrs = ['State: enabled',
                       'expr ptr'])

        self.runCmd('target stop-hook disable')

        self.expect('target stop-hook list', 'Stop Hook disabled successfully',
            substrs = ['State: disabled',
                       'expr ptr'])

        self.runCmd('target stop-hook enable')

        self.expect('target stop-hook list', 'Stop Hook enabled successfully',
            substrs = ['State: enabled',
                       'expr ptr'])

        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

        self.runCmd('target stop-hook delete')

        self.expect('target stop-hook list', 'Stop Hook deleted successfully',
            substrs = ['No stop hooks.'])
