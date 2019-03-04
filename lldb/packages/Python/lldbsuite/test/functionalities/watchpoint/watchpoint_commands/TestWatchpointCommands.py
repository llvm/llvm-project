"""
Test watchpoint list, enable, disable, and delete commands.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        self.line2 = line_number(
            self.source,
            '// Set 2nd break point for disable_then_enable test case.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source,
                                '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test
        # method.
        self.exe_name = self.testMethodName
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    @expectedFailureNetBSD
    def test_rw_watchpoint(self):
        """Test read_write watchpoint and expect to stop two times."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect(
            "watchpoint set variable -w read_write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = rw',
                '%s:%d' %
                (self.source,
                 self.decl)])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['Number of supported hardware watchpoints:',
                             'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
                    substrs=['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 2.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 2'])

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    def test_rw_watchpoint_delete(self):
        """Test delete watchpoint and expect not to stop for watchpoint."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect(
            "watchpoint set variable -w read_write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = rw',
                '%s:%d' %
                (self.source,
                 self.decl)])

        # Delete the watchpoint immediately, but set auto-confirm to true
        # first.
        self.runCmd("settings set auto-confirm true")
        self.expect("watchpoint delete",
                    substrs=['All watchpoints removed.'])
        # Restore the original setting of auto-confirm.
        self.runCmd("settings clear auto-confirm")

        # Use the '-v' option to do verbose listing of the watchpoint.
        self.runCmd("watchpoint list -v")

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
                    substrs=['exited'])

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    @expectedFailureNetBSD
    def test_rw_watchpoint_set_ignore_count(self):
        """Test watchpoint ignore count and expect to not to stop at all."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect(
            "watchpoint set variable -w read_write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = rw',
                '%s:%d' %
                (self.source,
                 self.decl)])

        # Set the ignore count of the watchpoint immediately.
        self.expect("watchpoint ignore -i 2",
                    substrs=['All watchpoints ignored.'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find an ignore_count of 2.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 0', 'ignore_count = 2'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
                    substrs=['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find a hit_count of 2 as well.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 2', 'ignore_count = 2'])

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    @expectedFailureNetBSD
    def test_rw_disable_after_first_stop(self):
        """Test read_write watchpoint but disable it after the first stop."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect(
            "watchpoint set variable -w read_write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = rw',
                '%s:%d' %
                (self.source,
                 self.decl)])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['state = enabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        # Before continuing, we'll disable the watchpoint, which means we won't
        # stop again after this.
        self.runCmd("watchpoint disable")

        self.expect("watchpoint list -v",
                    substrs=['state = disabled', 'hit_count = 1'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
                    substrs=['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 1.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 1'])

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    # Read-write watchpoints not supported on SystemZ
    @expectedFailureAll(archs=['s390x'])
    @expectedFailureNetBSD
    def test_rw_disable_then_enable(self):
        """Test read_write watchpoint, disable initially, then enable it."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line2, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect(
            "watchpoint set variable -w read_write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = rw',
                '%s:%d' %
                (self.source,
                 self.decl)])

        # Immediately, we disable the watchpoint.  We won't be stopping due to a
        # watchpoint after this.
        self.runCmd("watchpoint disable")

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['state = disabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the breakpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stop reason = breakpoint'])

        # Before continuing, we'll enable the watchpoint, which means we will
        # stop again after this.
        self.runCmd("watchpoint enable")

        self.expect("watchpoint list -v",
                    substrs=['state = enabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
                    substrs=['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 1.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 1'])
