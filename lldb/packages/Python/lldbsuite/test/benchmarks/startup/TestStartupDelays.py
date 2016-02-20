"""Test lldb's startup delays creating a target, setting a breakpoint, and run to breakpoint stop."""

from __future__ import print_function



import os, sys
import lldb
from lldbsuite.test import configuration
from lldbsuite.test import lldbtest_config
from lldbsuite.test.lldbbench import *

class StartupDelaysBench(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        BenchBase.setUp(self)
        # Create self.stopwatch2 for measuring "set first breakpoint".
        # The default self.stopwatch is for "create fresh target".
        self.stopwatch2 = Stopwatch()
        # Create self.stopwatch3 for measuring "run to breakpoint".
        self.stopwatch3 = Stopwatch()
        self.exe = lldbtest_config.lldbExec
        self.break_spec = '-n main'
        self.count = 30

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_startup_delay(self):
        """Test start up delays creating a target, setting a breakpoint, and run to breakpoint stop."""
        print()
        self.run_startup_delays_bench(self.exe, self.break_spec, self.count)
        print("lldb startup delay (create fresh target) benchmark:", self.stopwatch)
        print("lldb startup delay (set first breakpoint) benchmark:", self.stopwatch2)
        print("lldb startup delay (run to breakpoint) benchmark:", self.stopwatch3)

    def run_startup_delays_bench(self, exe, break_spec, count):
        import pexpect
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # Reset the stopwatchs now.
        self.stopwatch.reset()
        self.stopwatch2.reset()
        for i in range(count):
            # So that the child gets torn down after the test.
            self.child = pexpect.spawn('%s %s' % (lldbtest_config.lldbExec, self.lldbOption))
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            with self.stopwatch:
                # Create a fresh target.
                child.sendline('file %s' % exe) # Aka 'target create'.
                child.expect_exact(prompt)

            with self.stopwatch2:
                # Read debug info and set the first breakpoint.
                child.sendline('breakpoint set %s' % break_spec)
                child.expect_exact(prompt)

            with self.stopwatch3:
                # Run to the breakpoint just set.
                child.sendline('run')
                child.expect_exact(prompt)

            child.sendline('quit')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        # The test is about to end and if we come to here, the child process has
        # been terminated.  Mark it so.
        self.child = None
