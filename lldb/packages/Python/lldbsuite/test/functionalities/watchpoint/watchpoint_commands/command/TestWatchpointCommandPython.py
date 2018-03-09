"""
Test 'watchpoint command'.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointPythonCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source,
                                '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test
        # method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    @skipIfFreeBSD  # timing out on buildbot
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["aarch64"],
        bugnumber="llvm.org/pr27710")
    def test_watchpoint_command(self):
        """Test 'watchpoint command'."""
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

        # Now let's set a write-type watchpoint for 'global'.
        self.expect(
            "watchpoint set variable -w write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = w',
                '%s:%d' %
                (self.source,
                 self.decl)])

        self.runCmd(
            'watchpoint command add -s python 1 -o \'frame.EvaluateExpression("cookie = 777")\'')

        # List the watchpoint command we just added.
        self.expect("watchpoint command list 1",
                    substrs=['frame.EvaluateExpression', 'cookie = 777'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        # Check that the watchpoint snapshoting mechanism is working.
        self.expect("watchpoint list -v",
                    substrs=['old value:', ' = 0',
                             'new value:', ' = 1'])

        # The watchpoint command "forced" our global variable 'cookie' to
        # become 777.
        self.expect("frame variable --show-globals cookie",
                    substrs=['(int32_t)', 'cookie = 777'])

    @skipIfFreeBSD  # timing out on buildbot
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["aarch64"],
        bugnumber="llvm.org/pr27710")
    def test_continue_in_watchpoint_command(self):
        """Test continue in a watchpoint command."""
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

        # Now let's set a write-type watchpoint for 'global'.
        self.expect(
            "watchpoint set variable -w write global",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = w',
                '%s:%d' %
                (self.source,
                 self.decl)])

        cmd_script_file = os.path.join(self.getSourceDir(),
                                       "watchpoint_command.py")
        self.runCmd("command script import '%s'" % (cmd_script_file))

        self.runCmd(
            'watchpoint command add -F watchpoint_command.watchpoint_command')

        # List the watchpoint command we just added.
        self.expect("watchpoint command list 1",
                    substrs=['watchpoint_command.watchpoint_command'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
                    substrs=['stop reason = watchpoint'])

        # We should have hit the watchpoint once, set cookie to 888, then continued to the
        # second hit and set it to 999
        self.expect("frame variable --show-globals cookie",
                    substrs=['(int32_t)', 'cookie = 999'])

