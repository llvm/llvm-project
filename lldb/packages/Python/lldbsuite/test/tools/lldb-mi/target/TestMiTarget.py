"""
Test lldb-mi -target-xxx commands.
"""

from __future__ import print_function

import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiTargetTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # cannot attach to process on linux
    def test_lldbmi_target_attach_wait_for(self):
        """Test that 'lldb-mi --interpreter' works for -target-attach -n <name> --waitfor."""

        # Build target executable with unique name
        exeName = self.testMethodName
        d = {'EXE': exeName}
        self.buildProgram("test_attach.cpp", exeName)
        self.addTearDownCleanup(dictionary=d)

        self.spawnLldbMi(args=None)

        # Load executable
        # FIXME: -file-exec-and-sybmols is not required for target attach, but
        # the test will not pass without this
        self.runCmd("-file-exec-and-symbols %s" % exeName)
        self.expect("\^done")

        # Set up attach
        self.runCmd("-target-attach -n %s --waitfor" % exeName)
        time.sleep(4)  # Give attach time to setup

        # Start target process
        self.spawnSubprocess(os.path.join(os.path.dirname(__file__), exeName))
        self.addTearDownHook(self.cleanupSubprocesses)
        self.expect("\^done")

        # Set breakpoint on printf
        line = line_number('test_attach.cpp', '// BP_i++')
        self.runCmd("-break-insert -f test_attach.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")

        # Continue to breakpoint
        self.runCmd("-exec-continue")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Detach
        self.runCmd("-target-detach")
        self.expect("\^done")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # cannot attach to process on linux
    def test_lldbmi_target_attach_name(self):
        """Test that 'lldb-mi --interpreter' works for -target-attach -n <name>."""

        # Build target executable with unique name
        exeName = self.testMethodName
        d = {'EXE': exeName}
        self.buildProgram("test_attach.cpp", exeName)
        self.addTearDownCleanup(dictionary=d)

        # Start target process
        targetProcess = self.spawnSubprocess(
            os.path.join(os.path.dirname(__file__), exeName))
        self.addTearDownHook(self.cleanupSubprocesses)

        self.spawnLldbMi(args=None)

        # Set up atatch
        self.runCmd("-target-attach -n %s" % exeName)
        self.expect("\^done")

        # Set breakpoint on printf
        line = line_number('test_attach.cpp', '// BP_i++')
        self.runCmd("-break-insert -f test_attach.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")

        # Continue to breakpoint
        self.runCmd("-exec-continue")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Detach
        self.runCmd("-target-detach")
        self.expect("\^done")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # cannot attach to process on linux
    def test_lldbmi_target_attach_pid(self):
        """Test that 'lldb-mi --interpreter' works for -target-attach <pid>."""

        # Build target executable with unique name
        exeName = self.testMethodName
        d = {'EXE': exeName}
        self.buildProgram("test_attach.cpp", exeName)
        self.addTearDownCleanup(dictionary=d)

        # Start target process
        targetProcess = self.spawnSubprocess(
            os.path.join(os.path.dirname(__file__), exeName))
        self.addTearDownHook(self.cleanupSubprocesses)

        self.spawnLldbMi(args=None)

        # Set up atatch
        self.runCmd("-target-attach %d" % targetProcess.pid)
        self.expect("\^done")

        # Set breakpoint on printf
        line = line_number('test_attach.cpp', '// BP_i++')
        self.runCmd("-break-insert -f test_attach.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")

        # Continue to breakpoint
        self.runCmd("-exec-continue")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Detach
        self.runCmd("-target-detach")
        self.expect("\^done")
