"""
Test lldb-mi -gdb-set and -gdb-show commands.
"""

from __future__ import print_function


import unittest2
import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiGdbSetShowTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_set_target_async_default(self):
        """Test that 'lldb-mi --interpreter' switches to async mode by default."""

        self.spawnLldbMi(args=None)

        # Switch to sync mode
        self.runCmd("-gdb-set target-async off")
        self.expect("\^done")
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"off\"")

        # Test that -gdb-set switches to async by default
        self.runCmd("-gdb-set target-async")
        self.expect("\^done")
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"on\"")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFlakeyLinux("llvm.org/pr26028")  # Fails in ~1% of cases
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_set_target_async_on(self):
        """Test that 'lldb-mi --interpreter' can execute commands in async mode."""

        self.spawnLldbMi(args=None)

        # Switch to sync mode
        self.runCmd("-gdb-set target-async off")
        self.expect("\^done")
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"off\"")

        # Test that -gdb-set can switch to async mode
        self.runCmd("-gdb-set target-async on")
        self.expect("\^done")
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"on\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Test that program is executed in async mode
        self.runCmd("-exec-run")
        self.expect("\*running")
        self.expect("@\"argc=1")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="Failing in ~11/600 dosep runs (build 3120-3122)")
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_set_target_async_off(self):
        """Test that 'lldb-mi --interpreter' can execute commands in sync mode."""

        self.spawnLldbMi(args=None)

        # Test that -gdb-set can switch to sync mode
        self.runCmd("-gdb-set target-async off")
        self.expect("\^done")
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"off\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Test that program is executed in async mode
        self.runCmd("-exec-run")
        unexpected = ["\*running"]  # "\*running" is async notification
        it = self.expect(unexpected + ["@\"argc=1\\\\r\\\\n"])
        if it < len(unexpected):
            self.fail("unexpected found: %s" % unexpected[it])

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_show_target_async(self):
        """Test that 'lldb-mi --interpreter' in async mode by default."""

        self.spawnLldbMi(args=None)

        # Test that default target-async value is "on"
        self.runCmd("-gdb-show target-async")
        self.expect("\^done,value=\"on\"")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_show_language(self):
        """Test that 'lldb-mi --interpreter' can get current language."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -gdb-show language gets current language
        self.runCmd("-gdb-show language")
        self.expect("\^done,value=\"c\+\+\"")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @unittest2.expectedFailure("-gdb-set ignores unknown properties")
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_set_unknown(self):
        """Test that 'lldb-mi --interpreter' fails when setting an unknown property."""

        self.spawnLldbMi(args=None)

        # Test that -gdb-set fails if property is unknown
        self.runCmd("-gdb-set unknown some_value")
        self.expect("\^error")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @unittest2.expectedFailure("-gdb-show ignores unknown properties")
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_show_unknown(self):
        """Test that 'lldb-mi --interpreter' fails when showing an unknown property."""

        self.spawnLldbMi(args=None)

        # Test that -gdb-show fails if property is unknown
        self.runCmd("-gdb-show unknown")
        self.expect("\^error")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_gdb_set_ouptut_radix(self):
        """Test that 'lldb-mi --interpreter' works for -gdb-set output-radix."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_printf
        line = line_number('main.cpp', '// BP_printf')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Setup variable
        self.runCmd("-var-create var_a * a")
        self.expect(
            "\^done,name=\"var_a\",numchild=\"0\",value=\"10\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Test default output
        self.runCmd("-var-evaluate-expression var_a")
        self.expect("\^done,value=\"10\"")

        # Test hex output
        self.runCmd("-gdb-set output-radix 16")
        self.expect("\^done")
        self.runCmd("-var-evaluate-expression var_a")
        self.expect("\^done,value=\"0xa\"")

        # Test octal output
        self.runCmd("-gdb-set output-radix 8")
        self.expect("\^done")
        self.runCmd("-var-evaluate-expression var_a")
        self.expect("\^done,value=\"012\"")

        # Test decimal output
        self.runCmd("-gdb-set output-radix 10")
        self.expect("\^done")
        self.runCmd("-var-evaluate-expression var_a")
        self.expect("\^done,value=\"10\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfDarwin   # pexpect is known to be unreliable on Darwin
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @expectedFailureAll(
        bugnumber="llvm.org/pr31485: data-disassemble doesn't follow flavor settings")
    def test_lldbmi_gdb_set_disassembly_flavor(self):
        """Test that 'lldb-mi --interpreter' works for -gdb-set disassembly-flavor."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to BP_printf
        line = line_number('main.cpp', '// BP_printf')
        self.runCmd("-break-insert main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\".+addr=\"(0x[0-9a-f]+)\"")

        # Get starting and ending address from $pc
        pc = int(self.child.match.group(1), base=16)
        s_addr, e_addr = pc, pc + 1

        # Test default output (att)
        self.runCmd("-data-disassemble -s %d -e %d -- 0" % (s_addr, e_addr))
        self.expect("movl ")

        # Test intel style
        self.runCmd("-gdb-set disassembly-flavor intel")
        self.expect("\^done")
        self.runCmd("-data-disassemble -s %d -e %d -- 0" % (s_addr, e_addr))
        self.expect("mov ")

        # Test AT&T style
        self.runCmd("-gdb-set disassembly-flavor intel")
        self.expect("\^done")
        self.runCmd("-data-disassemble -s %d -e %d -- 0" % (s_addr, e_addr))
        self.expect("movl ")
