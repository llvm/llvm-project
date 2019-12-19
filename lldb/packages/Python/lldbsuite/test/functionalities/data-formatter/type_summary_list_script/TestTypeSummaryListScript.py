"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeSummaryListScriptTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_typesummarylist_script(self):
        """Test data formatter commands."""
        self.build()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', 'Break here')

    def data_formatter_commands(self):
        """Test printing out Python summary formatters."""
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
            self.runCmd('type category delete TSLSFormatters', check=False)
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)

        self.addTearDownHook(cleanup)

        self.runCmd("command script import tslsformatters.py")

        self.expect(
            "frame variable myStruct",
            substrs=['A data formatter at work'])

        self.expect('type summary list', substrs=['Struct_SummaryFormatter'])
        self.expect(
            'type summary list Struct',
            substrs=['Struct_SummaryFormatter'])
