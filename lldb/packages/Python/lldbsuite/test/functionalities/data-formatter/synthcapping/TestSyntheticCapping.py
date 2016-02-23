"""
Check for an issue where capping does not work because the Target pointer appears to be changing behind our backs
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class SyntheticCappingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Check for an issue where capping does not work because the Target pointer appears to be changing behind our backs."""
        self.build()
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
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # set up the synthetic children provider
        self.runCmd("script from fooSynthProvider import *")
        self.runCmd("type synth add -l fooSynthProvider foo")

        # check that the synthetic children work, so we know we are doing the right thing
        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777216',
                               'a = 0']);

        # check that capping works
        self.runCmd("settings set target.max-children-count 2", check=False)
        
        self.expect("frame variable f00_1",
                    substrs = ['...',
                               'fake_a = 16777216',
                               'a = 0']);
        
        self.expect("frame variable f00_1", matching=False,
                    substrs = ['r = 33']);

        
        self.runCmd("settings set target.max-children-count 256", check=False)

        self.expect("frame variable f00_1", matching=True,
                    substrs = ['r = 33']);
