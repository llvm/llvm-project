"""
Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class Radar12481949DataFormatterTestCase(TestBase):

    # test for rdar://problem/12481949
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

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
            self.runCmd('type format delete hex', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsSigned() == -1,
            "GetValueAsSigned() says -1")
        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsSigned() != 0xFFFFFFFF,
            "GetValueAsSigned() does not say 0xFFFFFFFF")
        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsSigned() != 0xFFFFFFFFFFFFFFFF,
            "GetValueAsSigned() does not say 0xFFFFFFFFFFFFFFFF")

        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsUnsigned() != -1,
            "GetValueAsUnsigned() does not say -1")
        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsUnsigned() == 0xFFFFFFFF,
            "GetValueAsUnsigned() says 0xFFFFFFFF")
        self.assertTrue(
            self.frame().FindVariable("myvar").GetValueAsUnsigned() != 0xFFFFFFFFFFFFFFFF,
            "GetValueAsUnsigned() does not says 0xFFFFFFFFFFFFFFFF")
