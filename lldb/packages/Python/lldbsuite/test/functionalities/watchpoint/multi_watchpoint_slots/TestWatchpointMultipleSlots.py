"""
Test watchpoint slots we should not be able to install multiple watchpoints
within same word boundary. We should be able to install individual watchpoints
on any of the bytes, half-word, or word. This is only for ARM/AArch64 targets.
"""

from __future__ import print_function

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointSlotsTestCase(TestBase):
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

    # This is a arm and aarch64 specific test case. No other architectures tested.
    @skipIf(archs=no_match(['arm', 'aarch64']))
    def test_multiple_watchpoints_on_same_word(self):

        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Detect line number after which we are going to increment arrayName.
        loc_line = line_number('main.c', '// About to write byteArray')

        # Set a breakpoint on the line detected above.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", loc_line, num_expected_locations=1, loc_exact=True)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                     substrs=['stopped', 'stop reason = breakpoint'])

        # Delete breakpoint we just hit.
        self.expect("breakpoint delete 1", substrs=['1 breakpoints deleted'])

        # Set a watchpoint at byteArray[0]
        self.expect("watchpoint set variable byteArray[0]", WATCHPOINT_CREATED,
                    substrs=['Watchpoint created','size = 1'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v 1", substrs=['hit_count = 0'])

        # debugserver on ios doesn't give an error, it creates another watchpoint,
        # only expect errors on non-darwin platforms.
        if not self.platformIsDarwin():
            # Try setting a watchpoint at byteArray[1]
            self.expect("watchpoint set variable byteArray[1]", error=True,
                        substrs=['Watchpoint creation failed'])

        self.runCmd("process continue")

        # We should be stopped due to the watchpoint.
        # The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stopped', 'stop reason = watchpoint 1'])

        # Delete the watchpoint we hit above successfully.
        self.expect("watchpoint delete 1", substrs=['1 watchpoints deleted'])

        # Set a watchpoint at byteArray[3]
        self.expect("watchpoint set variable byteArray[3]", WATCHPOINT_CREATED,
                    substrs=['Watchpoint created','size = 1'])
   
        # Resume inferior.
        self.runCmd("process continue")

        # We should be stopped due to the watchpoint.
        # The stop reason of the thread should be watchpoint.
        if self.platformIsDarwin():
            # On darwin we'll hit byteArray[3] which is watchpoint 2
            self.expect("thread list -v", STOPPED_DUE_TO_WATCHPOINT,
                        substrs=['stopped', 'stop reason = watchpoint 2'])
        else:
            self.expect("thread list -v", STOPPED_DUE_TO_WATCHPOINT,
                        substrs=['stopped', 'stop reason = watchpoint 3'])
   
        # Resume inferior.
        self.runCmd("process continue")
