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

class SyntheticFilterRecomputingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    @skipUnlessDarwin
    def test_rdar12437442_with_run_command(self):
        """Test that we update SBValues correctly as dynamic types change."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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

        # Now run the bulk of the test
        id_x = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable("x")
        id_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        id_x.SetPreferSyntheticValue(True)
        
        if self.TraceOn():
            self.runCmd("expr --dynamic-type run-target --ptr-depth 1 -- x")

        self.assertTrue(id_x.GetSummary() == '@"5 elements"', "array does not get correct summary")

        self.runCmd("next")
        self.runCmd("frame select 0")

        id_x = self.dbg.GetSelectedTarget().GetProcess().GetSelectedThread().GetSelectedFrame().FindVariable("x")
        id_x.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        id_x.SetPreferSyntheticValue(True)

        if self.TraceOn():
            self.runCmd("expr --dynamic-type run-target --ptr-depth 1 -- x")

        self.assertTrue(id_x.GetNumChildren() == 7, "dictionary does not have 7 children")
        id_x.SetPreferSyntheticValue(False)
        self.assertFalse(id_x.GetNumChildren() == 7, "dictionary still looks synthetic")
        id_x.SetPreferSyntheticValue(True)
        self.assertTrue(id_x.GetSummary() == "7 key/value pairs", "dictionary does not get correct summary")
