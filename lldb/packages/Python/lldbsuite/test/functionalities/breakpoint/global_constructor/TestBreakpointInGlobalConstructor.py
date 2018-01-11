"""
Test that we can hit breakpoints in global constructors
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointInGlobalConstructors(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.line_foo = line_number('foo.cpp', '// !BR_foo')
        self.line_main = line_number('main.cpp', '// !BR_main')

    @expectedFailureAll(bugnumber="llvm.org/pr35480", oslist=["linux"])
    def test(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file %s" % exe)

        bp_main = lldbutil.run_break_set_by_file_and_line(
            self, 'main.cpp', self.line_main)
        bp_foo = lldbutil.run_break_set_by_file_and_line(
            self, 'foo.cpp', self.line_foo)

        self.runCmd("run")

        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(
                self.process(), bp_foo))

        self.runCmd("continue")

        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(
                self.process(), bp_main))
