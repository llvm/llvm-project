"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class GlobalsDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"JustATest\" Point")

        # Simply check we can get at global variables
        self.expect("target variable g_point",
                    substrs=['JustATest'])

        self.expect("target variable g_point_pointer",
                    substrs=['(Point *) g_point_pointer ='])

        # Print some information about the variables
        # (we ignore the actual values)
        self.runCmd(
            "type summary add --summary-string \"(x=${var.x},y=${var.y})\" Point")

        self.expect("target variable g_point",
                    substrs=['x=',
                             'y='])

        self.expect("target variable g_point_pointer",
                    substrs=['(Point *) g_point_pointer ='])

        # Test Python code on resulting SBValue
        self.runCmd(
            "type summary add --python-script \"return 'x=' + str(valobj.GetChildMemberWithName('x').GetValue());\" Point")

        self.expect("target variable g_point",
                    substrs=['x='])

        self.expect("target variable g_point_pointer",
                    substrs=['(Point *) g_point_pointer ='])
