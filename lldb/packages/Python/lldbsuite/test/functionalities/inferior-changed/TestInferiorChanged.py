"""Test lldb reloads the inferior after it was changed during the session."""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil

class ChangedInferiorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfHostWindows
    def test_inferior_crashing(self):
        """Test lldb reloads the inferior after it was changed during the session."""
        self.build()
        self.inferior_crashing()
        self.cleanup()
        # lldb needs to recognize the inferior has changed. If lldb needs to check the
        # new module timestamp, make sure it is not the same as the old one, so add a
        # 1 second delay.
        time.sleep(1)
        d = {'C_SOURCES': 'main2.c'}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.inferior_not_crashing()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line1 = line_number('main.c', '// Crash here.')
        self.line2 = line_number('main2.c', '// Not crash here.')

    def inferior_crashing(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        self.exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + self.exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        # We should have one crashing thread
        self.assertEqual(
                len(lldbutil.get_crashed_threads(self, self.dbg.GetSelectedTarget().GetProcess())),
                1,
                STOPPED_DUE_TO_EXC_BAD_ACCESS)

        # And it should report the correct line number.
        self.expect("thread backtrace all", substrs = ['main.c:%d' % self.line1])

    def inferior_not_crashing(self):
        """Test lldb reloads the inferior after it was changed during the session."""
        self.runCmd("process kill")
        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("process status")

        self.assertNotEqual(
                len(lldbutil.get_crashed_threads(self, self.dbg.GetSelectedTarget().GetProcess())),
                1,
                "Inferior changed, but lldb did not perform a reload")

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main2.c", self.line2, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.runCmd("frame variable int_ptr")
        self.expect("frame variable *int_ptr",
            substrs = ['= 7'])
        self.expect("expression *int_ptr",
            substrs = ['= 7'])
