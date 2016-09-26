"""
Test lldb ability to unwind a stack with a function containing a call to the
'__builtin_trap' intrinsic, which GCC (4.6) encodes to an illegal opcode.
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BuiltinTrapTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    # gcc generates incorrect linetable
    @expectedFailureAll(archs="arm", compiler="gcc", triple=".*-android")
    @expectedFailureAll(oslist=['linux'], archs=['arm'])
    @skipIfWindows
    def test_with_run_command(self):
        """Test that LLDB handles a function with __builtin_trap correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line,
                                                num_expected_locations=1,
                                                loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # print backtrace, expect both 'bar' and 'main' functions to be listed
        self.expect('bt', substrs=['bar', 'main'])

        # go up one frame
        self.runCmd("up", RUN_SUCCEEDED)

        # evaluate a local
        self.expect('p foo', substrs=['= 5'])
