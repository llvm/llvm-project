"""
Test lldb-mi -exec-xxx commands.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiExecTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_abort(self):
        """Test that 'lldb-mi --interpreter' works for -exec-abort."""

        self.spawnLldbMi(args=None)

        # Test that -exec-abort fails on invalid process
        self.runCmd("-exec-abort")
        self.expect(
            "\^error,msg=\"Command 'exec-abort'\. Invalid process during debug session\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments arg1")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

        # Test that program can be run again
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed again
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted again
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_arguments_set(self):
        """Test that 'lldb-mi --interpreter' can pass args using -exec-arguments."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd(
            "-exec-arguments --arg1 \"2nd arg\" third_arg fourth=\"4th arg\"")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc and argv to see if arg passed
        # Note that exactly=True is needed to avoid extra escaping for re
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"5\"")
        #self.runCmd("-data-evaluate-expression argv[1]")
        # self.expect("\^done,value=\"--arg1\"")
        self.runCmd("-interpreter-exec command \"print argv[1]\"")
        self.expect("\\\"--arg1\\\"", exactly=True)
        #self.runCmd("-data-evaluate-expression argv[2]")
        #self.expect("\^done,value=\"2nd arg\"")
        self.runCmd("-interpreter-exec command \"print argv[2]\"")
        self.expect("\\\"2nd arg\\\"", exactly=True)
        #self.runCmd("-data-evaluate-expression argv[3]")
        # self.expect("\^done,value=\"third_arg\"")
        self.runCmd("-interpreter-exec command \"print argv[3]\"")
        self.expect("\\\"third_arg\\\"", exactly=True)
        #self.runCmd("-data-evaluate-expression argv[4]")
        #self.expect("\^done,value=\"fourth=\\\\\\\"4th arg\\\\\\\"\"")
        self.runCmd("-interpreter-exec command \"print argv[4]\"")
        self.expect("\\\"fourth=\\\\\\\"4th arg\\\\\\\"\\\"", exactly=True)

    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_arguments_reset(self):
        """Test that 'lldb-mi --interpreter' can reset previously set args using -exec-arguments."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments arg1")
        self.expect("\^done")
        self.runCmd("-exec-arguments")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"1\"")
