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


class LibCxxAtomicTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def get_variable(self, name):
        var = self.frame().FindVariable(name)
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    @skipIf(compiler="gcc")
    @skipIfWindows  # libc++ not ported to Windows yet
    def test(self):
        """Test that std::atomic as defined by libc++ is correctly printed by LLDB"""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."))

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.skip_if_library_missing(
            self, self.target(), lldbutil.PrintableRegex("libc\+\+"))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        s = self.get_variable('s')
        i = self.get_variable('i')

        if self.TraceOn():
            print(s)
        if self.TraceOn():
            print(i)

        self.assertTrue(i.GetValueAsUnsigned(0) == 5, "i == 5")
        self.assertTrue(s.GetNumChildren() == 2, "s has two children")
        self.assertTrue(
            s.GetChildAtIndex(0).GetValueAsUnsigned(0) == 1,
            "s.x == 1")
        self.assertTrue(
            s.GetChildAtIndex(1).GetValueAsUnsigned(0) == 2,
            "s.y == 2")
