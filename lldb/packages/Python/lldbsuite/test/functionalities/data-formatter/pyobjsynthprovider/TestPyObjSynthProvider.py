"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function



import datetime
import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class PyObjectSynthProviderTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_print_array(self):
        """Test that expr -Z works"""
        self.build()
        self.provider_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', 'break here')

    def provider_data_formatter_commands(self):
        """Test that the PythonObjectSyntheticChildProvider helper class works"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        
        self.runCmd('command script import provider.py')
        self.runCmd('type synthetic add Foo --python-class provider.SyntheticChildrenProvider')
        self.expect('frame variable f.Name', substrs=['"Enrico"'])
        self.expect('frame variable f', substrs=['ID = 123456', 'Name = "Enrico"', 'Rate = 1.25'])
        self.expect('expression f', substrs=['ID = 123456', 'Name = "Enrico"', 'Rate = 1.25'])
