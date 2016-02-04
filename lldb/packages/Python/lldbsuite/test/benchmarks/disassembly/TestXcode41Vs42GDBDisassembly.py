"""Disassemble lldb's Driver::MainLoop() functions comparing Xcode 4.1 vs. 4.2's gdb."""

from __future__ import print_function



import os, sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbbench import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil

class XCode41Vs42GDBDisassembly(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        BenchBase.setUp(self)
        self.gdb_41_exe = '/Xcode41/usr/bin/gdb'
        self.gdb_42_exe = '/Developer/usr/bin/gdb'
        self.exe = lldbtest_config.lldbExec
        self.function = 'Driver::MainLoop()'
        self.gdb_41_avg = None
        self.gdb_42_avg = None
        self.count = 5

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_run_41_then_42(self):
        """Test disassembly on a large function with 4.1 vs. 4.2's gdb."""
        print()
        self.run_gdb_disassembly(self.gdb_41_exe, self.exe, self.function, self.count)
        print("4.1 gdb benchmark:", self.stopwatch)
        self.gdb_41_avg = self.stopwatch.avg()
        self.run_gdb_disassembly(self.gdb_42_exe, self.exe, self.function, self.count)
        print("4.2 gdb benchmark:", self.stopwatch)
        self.gdb_42_avg = self.stopwatch.avg()
        print("gdb_42_avg/gdb_41_avg: %f" % (self.gdb_42_avg/self.gdb_41_avg))

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_run_42_then_41(self):
        """Test disassembly on a large function with 4.1 vs. 4.2's gdb."""
        print()
        self.run_gdb_disassembly(self.gdb_42_exe, self.exe, self.function, self.count)
        print("4.2 gdb benchmark:", self.stopwatch)
        self.gdb_42_avg = self.stopwatch.avg()
        self.run_gdb_disassembly(self.gdb_41_exe, self.exe, self.function, self.count)
        print("4.1 gdb benchmark:", self.stopwatch)
        self.gdb_41_avg = self.stopwatch.avg()
        print("gdb_42_avg/gdb_41_avg: %f" % (self.gdb_42_avg/self.gdb_41_avg))

    def run_gdb_disassembly(self, gdb_exe_path, exe, function, count):
        import pexpect
        # Set self.child_prompt, which is "(gdb) ".
        self.child_prompt = '(gdb) '
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s --nx %s' % (gdb_exe_path, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        child.expect_exact(prompt)
        child.sendline('break %s' % function)
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)

        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            with self.stopwatch:
                # Disassemble the function.
                child.sendline('disassemble')
                child.expect_exact(prompt)
            child.sendline('next')
            child.expect_exact(prompt)

        child.sendline('quit')
        child.expect_exact('The program is running.  Exit anyway?')
        child.sendline('y')
        try:
            self.child.expect(pexpect.EOF)
        except:
            pass

        if self.TraceOn():
            print("gdb disassembly benchmark:", str(self.stopwatch))
        self.child = None
