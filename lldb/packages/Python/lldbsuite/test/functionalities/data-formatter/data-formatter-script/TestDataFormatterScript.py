"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ScriptDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """Test data formatter commands."""
        self.build()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
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
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Set the script here to ease the formatting
        script = 'a = valobj.GetChildMemberWithName(\'integer\'); a_val = a.GetValue(); str = \'Hello from Python, \' + a_val + \' time\'; return str + (\'!\' if a_val == \'1\' else \'s!\');'

        self.runCmd(
            "type summary add i_am_cool --python-script \"%s\"" %
            script)
        self.expect('type summary list i_am_cool', substrs=[script])

        self.expect("frame variable one",
                    substrs=['Hello from Python',
                             '1 time!'])

        self.expect("frame variable two",
                    substrs=['Hello from Python',
                             '4 times!'])

        self.runCmd("n")  # skip ahead to make values change

        self.expect("frame variable three",
                    substrs=['Hello from Python, 10 times!',
                             'Hello from Python, 4 times!'])

        self.runCmd("n")  # skip ahead to make values change

        self.expect("frame variable two",
                    substrs=['Hello from Python',
                             '1 time!'])

        script = 'a = valobj.GetChildMemberWithName(\'integer\'); a_val = a.GetValue(); str = \'int says \' + a_val; return str;'

        # Check that changes in the script are immediately reflected
        self.runCmd(
            "type summary add i_am_cool --python-script \"%s\"" %
            script)

        self.expect("frame variable two",
                    substrs=['int says 1'])

        self.expect("frame variable twoptr",
                    substrs=['int says 1'])

        # Change the summary
        self.runCmd(
            "type summary add --summary-string \"int says ${var.integer}, and float says ${var.floating}\" i_am_cool")

        self.expect("frame variable two",
                    substrs=['int says 1',
                             'and float says 2.71'])
        # Try it for pointers
        self.expect("frame variable twoptr",
                    substrs=['int says 1',
                             'and float says 2.71'])

        # Force a failure for pointers
        self.runCmd(
            "type summary add i_am_cool -p --python-script \"%s\"" %
            script)

        self.expect("frame variable twoptr", matching=False,
                    substrs=['and float says 2.71'])

        script = 'return \'Python summary\''

        self.runCmd(
            "type summary add --name test_summary --python-script \"%s\"" %
            script)

        # attach the Python named summary to someone
        self.expect("frame variable one --summary test_summary",
                    substrs=['Python summary'])

        # should not bind to the type
        self.expect("frame variable two", matching=False,
                    substrs=['Python summary'])

        # and should not stick to the variable
        self.expect("frame variable one", matching=False,
                    substrs=['Python summary'])

        self.runCmd(
            "type summary add i_am_cool --summary-string \"Text summary\"")

        # should be temporary only
        self.expect("frame variable one", matching=False,
                    substrs=['Python summary'])

        # use the type summary
        self.expect("frame variable two",
                    substrs=['Text summary'])

        self.runCmd("n")  # skip ahead to make values change

        # both should use the type summary now
        self.expect("frame variable one",
                    substrs=['Text summary'])

        self.expect("frame variable two",
                    substrs=['Text summary'])

        # disable type summary for pointers, and make a Python regex summary
        self.runCmd(
            "type summary add i_am_cool -p --summary-string \"Text summary\"")
        self.runCmd("type summary add -x cool --python-script \"%s\"" % script)

        # variables should stick to the type summary
        self.expect("frame variable one",
                    substrs=['Text summary'])

        self.expect("frame variable two",
                    substrs=['Text summary'])

        # array and pointer should match the Python one
        self.expect("frame variable twoptr",
                    substrs=['Python summary'])

        self.expect("frame variable array",
                    substrs=['Python summary'])

        # return pointers to the type summary
        self.runCmd(
            "type summary add i_am_cool --summary-string \"Text summary\"")

        self.expect("frame variable one",
                    substrs=['Text summary'])

        self.expect("frame variable two",
                    substrs=['Text summary'])

        self.expect("frame variable twoptr",
                    substrs=['Text summary'])

        self.expect("frame variable array",
                    substrs=['Python summary'])
