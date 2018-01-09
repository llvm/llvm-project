"""
Test watchpoint size cases (1-byte, 2-byte, 4-byte).
Make sure we can watch all bytes, words or double words individually
when they are packed in a 8-byte region.

"""

from __future__ import print_function

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointSizeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = 'a.out'
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    # Watchpoints not supported
    @expectedFailureAndroid(archs=['arm', 'aarch64'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    def test_byte_size_watchpoints_with_byte_selection(self):
        """Test to selectively watch different bytes in a 8-byte array."""
        self.run_watchpoint_size_test('byteArray', 8, '1')

    # Watchpoints not supported
    @expectedFailureAndroid(archs=['arm', 'aarch64'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    def test_two_byte_watchpoints_with_word_selection(self):
        """Test to selectively watch different words in an 8-byte word array."""
        self.run_watchpoint_size_test('wordArray', 4, '2')

    # Watchpoints not supported
    @expectedFailureAndroid(archs=['arm', 'aarch64'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    def test_four_byte_watchpoints_with_dword_selection(self):
        """Test to selectively watch two double words in an 8-byte dword array."""
        self.run_watchpoint_size_test('dwordArray', 2, '4')

    def run_watchpoint_size_test(self, arrayName, array_size, watchsize):
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Detect line number after which we are going to increment arrayName.
        loc_line = line_number('main.c', '// About to write ' + arrayName)

        # Set a breakpoint on the line detected above.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", loc_line, num_expected_locations=1, loc_exact=True)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        for i in range(array_size):
            # We should be stopped again due to the breakpoint.
            # The stop reason of the thread should be breakpoint.
            self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                        substrs=['stopped', 'stop reason = breakpoint'])

            # Set a read_write type watchpoint arrayName
            watch_loc = arrayName + "[" + str(i) + "]"
            self.expect(
                "watchpoint set variable -w read_write " +
                watch_loc,
                WATCHPOINT_CREATED,
                substrs=[
                    'Watchpoint created',
                    'size = ' +
                    watchsize,
                    'type = rw'])

            # Use the '-v' option to do verbose listing of the watchpoint.
            # The hit count should be 0 initially.
            self.expect("watchpoint list -v", substrs=['hit_count = 0'])

            self.runCmd("process continue")

            # We should be stopped due to the watchpoint.
            # The stop reason of the thread should be watchpoint.
            self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
                        substrs=['stopped', 'stop reason = watchpoint'])

            # Use the '-v' option to do verbose listing of the watchpoint.
            # The hit count should now be 1.
            self.expect("watchpoint list -v",
                        substrs=['hit_count = 1'])

            self.runCmd("process continue")

            # We should be stopped due to the watchpoint.
            # The stop reason of the thread should be watchpoint.
            self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
                        substrs=['stopped', 'stop reason = watchpoint'])

            # Use the '-v' option to do verbose listing of the watchpoint.
            # The hit count should now be 1.
            # Verify hit_count has been updated after value has been read.
            self.expect("watchpoint list -v",
                        substrs=['hit_count = 2'])

            # Delete the watchpoint immediately, but set auto-confirm to true
            # first.
            self.runCmd("settings set auto-confirm true")
            self.expect(
                "watchpoint delete",
                substrs=['All watchpoints removed.'])
            # Restore the original setting of auto-confirm.
            self.runCmd("settings clear auto-confirm")

            self.runCmd("process continue")
