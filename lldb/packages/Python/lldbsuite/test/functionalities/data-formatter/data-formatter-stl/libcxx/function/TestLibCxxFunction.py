"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxFunctionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def get_variable(self, name):
        var = self.frame().FindVariable(name)
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::function as defined by libc++ is correctly printed by LLDB"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."))

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        f1 = self.get_variable('f1')
        f2 = self.get_variable('f2')

        if self.TraceOn():
            print(f1)
        if self.TraceOn():
            print(f2)

        self.assertTrue(f1.GetValueAsUnsigned(0) != 0, 'f1 has a valid value')
        self.assertTrue(f2.GetValueAsUnsigned(0) != 0, 'f2 has a valid value')
