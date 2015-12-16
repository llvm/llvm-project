"""Disassemble lldb's Driver::MainLoop() functions comparing lldb against gdb."""

from __future__ import print_function



import os, sys
import lldb
from lldbsuite.test import configuration
from lldbsuite.test.lldbbench import *

def is_exe(fpath):
    """Returns true if fpath is an executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

class DisassembleDriverMainLoop(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        """
        Note that lldbtest_config.lldbExec can be specified with the LLDB_EXEC env variable (see
        dotest.py), and gdbExec can be specified with the GDB_EXEC env variable.
        This provides a flexibility in specifying different versions of gdb for
        comparison purposes.
        """
        BenchBase.setUp(self)
        # If env var GDB_EXEC is specified, use it; otherwise, use gdb in your
        # PATH env var.
        if "GDB_EXEC" in os.environ and is_exe(os.environ["GDB_EXEC"]):
            self.gdbExec = os.environ["GDB_EXEC"]
        else:
            self.gdbExec = "gdb"

        self.exe = lldbtest_config.lldbExec
        self.function = 'Driver::MainLoop()'
        self.lldb_avg = None
        self.gdb_avg = None
        self.count = 5

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_run_lldb_then_gdb(self):
        """Test disassembly on a large function with lldb vs. gdb."""
        print()
        print("lldb path: %s" % lldbtest_config.lldbExec)
        print("gdb path: %s" % self.gdbExec)

        print()
        self.run_lldb_disassembly(self.exe, self.function, self.count)
        print("lldb benchmark:", self.stopwatch)
        self.run_gdb_disassembly(self.exe, self.function, self.count)
        print("gdb benchmark:", self.stopwatch)
        print("lldb_avg/gdb_avg: %f" % (self.lldb_avg/self.gdb_avg))

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_run_gdb_then_lldb(self):
        """Test disassembly on a large function with lldb vs. gdb."""
        print()
        print("lldb path: %s" % lldbtest_config.lldbExec)
        print("gdb path: %s" % self.gdbExec)

        print()
        self.run_gdb_disassembly(self.exe, self.function, self.count)
        print("gdb benchmark:", self.stopwatch)
        self.run_lldb_disassembly(self.exe, self.function, self.count)
        print("lldb benchmark:", self.stopwatch)
        print("lldb_avg/gdb_avg: %f" % (self.lldb_avg/self.gdb_avg))

    def run_lldb_disassembly(self, exe, function, count):
        import pexpect
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (lldbtest_config.lldbExec, self.lldbOption, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        child.expect_exact(prompt)
        child.sendline('breakpoint set -F %s' % function)
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)

        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            with self.stopwatch:
                # Disassemble the function.
                child.sendline('disassemble -f')
                child.expect_exact(prompt)
            child.sendline('next')
            child.expect_exact(prompt)

        child.sendline('quit')
        try:
            self.child.expect(pexpect.EOF)
        except:
            pass

        self.lldb_avg = self.stopwatch.avg()
        if self.TraceOn():
            print("lldb disassembly benchmark:", str(self.stopwatch))
        self.child = None

    def run_gdb_disassembly(self, exe, function, count):
        import pexpect
        # Set self.child_prompt, which is "(gdb) ".
        self.child_prompt = '(gdb) '
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s --nx %s' % (self.gdbExec, exe))
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

        self.gdb_avg = self.stopwatch.avg()
        if self.TraceOn():
            print("gdb disassembly benchmark:", str(self.stopwatch))
        self.child = None
