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

        self.expect("frame variable f1",
                    substrs=['f1 =  Function = foo(int, int)'])

        self.expect("frame variable f2",
                    substrs=['f2 =  Lambda in File main.cpp at Line 26'])

        self.expect("frame variable f3",
                    substrs=['f3 =  Lambda in File main.cpp at Line 30'])

        self.expect("frame variable f4",
                    substrs=['f4 =  Function in File main.cpp at Line 16'])

        self.expect("frame variable f5",
                    substrs=['f5 =  Function = Bar::add_num(int) const'])
