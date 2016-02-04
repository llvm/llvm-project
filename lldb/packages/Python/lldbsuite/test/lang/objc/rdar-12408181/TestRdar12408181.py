"""
Test that we are able to find out how many children NSWindow has
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

@skipUnlessDarwin
class Rdar12408181TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    def test_nswindow_count(self):
        """Test that we are able to find out how many children NSWindow has."""
        d = {'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)

        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.main_source, self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        if self.frame().EvaluateExpression('(void*)_CGSDefaultConnection()').GetValueAsUnsigned() != 0:
            window = self.frame().FindVariable("window")
            window_dynamic = window.GetDynamicValue(lldb.eDynamicCanRunTarget)
            self.assertTrue(window.GetNumChildren() > 1, "NSWindow (static) only has 1 child!")
            self.assertTrue(window_dynamic.GetNumChildren() > 1, "NSWindow (dynamic) only has 1 child!")
            self.assertTrue(window.GetChildAtIndex(0).IsValid(), "NSWindow (static) has an invalid child")
            self.assertTrue(window_dynamic.GetChildAtIndex(0).IsValid(), "NSWindow (dynamic) has an invalid child")
        else:
            self.skipTest('no WindowServer connection')
