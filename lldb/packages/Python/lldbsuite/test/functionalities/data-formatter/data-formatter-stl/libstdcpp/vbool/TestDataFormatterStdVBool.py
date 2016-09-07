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


class StdVBoolDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @expectedFailureAll(
        oslist=['freebsd'],
        bugnumber='llvm.org/pr20548 fails to build on lab.llvm.org buildbot')
    @expectedFailureAll(
        compiler="icc",
        bugnumber="llvm.org/pr15301 LLDB prints incorrect sizes of STL containers")
    @skipIfWindows  # libstdcpp not ported to Windows.
    @skipIfDarwin
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)

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
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "frame variable vBool",
            substrs=[
                'size=49',
                '[0] = false',
                '[1] = true',
                '[18] = false',
                '[27] = true',
                '[36] = false',
                '[47] = true',
                '[48] = true'])

        self.expect(
            "expr vBool",
            substrs=[
                'size=49',
                '[0] = false',
                '[1] = true',
                '[18] = false',
                '[27] = true',
                '[36] = false',
                '[47] = true',
                '[48] = true'])
