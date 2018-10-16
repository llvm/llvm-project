"""
Test hardware breakpoints for multiple threads.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# Hardware breakpoints are supported only by platforms mentioned in oslist.
@skipUnlessPlatform(oslist=['linux'])
class HardwareBreakpointMultiThreadTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    # LLDB supports hardware breakpoints for arm and aarch64 architectures.
    @skipIf(archs=no_match(['arm', 'aarch64']))
    @expectedFailureAndroid
    def test_hw_break_set_delete_multi_thread(self):
        self.build()
        self.setTearDownCleanup()
        self.break_multi_thread('delete')

    # LLDB supports hardware breakpoints for arm and aarch64 architectures.
    @skipIf(archs=no_match(['arm', 'aarch64']))
    @expectedFailureAndroid
    def test_hw_break_set_disable_multi_thread(self):
        self.build()
        self.setTearDownCleanup()
        self.break_multi_thread('disable')

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.first_stop = line_number(
            self.source, 'Starting thread creation with hardware breakpoint set')

    def break_multi_thread(self, removal_type):
        """Test that lldb hardware breakpoints work for multiple threads."""
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)

        # Stop in main before creating any threads.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.first_stop, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now set a hardware breakpoint in thread function.
        self.expect("breakpoint set -b hw_break_function --hardware",
            substrs=[
                'Breakpoint',
                'hw_break_function',
                'address = 0x'])

        # We should stop in hw_break_function function for 4 threads.
        count = 0

        while count < 2 :

            self.runCmd("process continue")

            # We should be stopped in hw_break_function
            # The stop reason of the thread should be breakpoint.
            self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                substrs=[
                    'stop reason = breakpoint',
                    'hw_break_function'])

            # Continue the loop and test that we are stopped 4 times.
            count += 1

        if removal_type == 'delete':
            self.runCmd("settings set auto-confirm true")

            # Now 'breakpoint delete' should just work fine without confirmation
            # prompt from the command interpreter.
            self.expect("breakpoint delete",
                        startstr="All breakpoints removed")

            # Restore the original setting of auto-confirm.
            self.runCmd("settings clear auto-confirm")

        elif removal_type == 'disable':
            self.expect("breakpoint disable",
                        startstr="All breakpoints disabled.")

        # Continue. Program should exit without stopping anywhere.
        self.runCmd("process continue")

        # Process should have stopped and exited with status = 0
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* exited with status = 0'])
