"""
Test that the lldb-mi handles signals properly.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiSignalTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Fails on FreeBSD apparently due to thread race conditions
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_stopped_when_interrupt(self):
        """Test that 'lldb-mi --interpreter' interrupt and resume a looping app."""

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

        # Set doloop=1 and run (to loop forever)
        self.runCmd("-data-evaluate-expression \"do_loop=1\"")
        self.expect("\^done,value=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")

        # Test that -exec-interrupt can interrupt an execution
        self.runCmd("-exec-interrupt")
        self.expect(
            "\*stopped,reason=\"signal-received\",signal-name=\"SIGINT\",signal-meaning=\"Interrupt\",.+?thread-id=\"1\",stopped-threads=\"all\"")

        # Continue (to loop forever)
        self.runCmd("-exec-continue")
        self.expect("\^running")

        # Test that Ctrl+C can interrupt an execution
        self.child.sendintr()  # FIXME: here uses self.child directly
        self.expect(
            "\*stopped,reason=\"signal-received\",signal-name=\"SIGINT\",signal-meaning=\"Interrupt\",.*thread-id=\"1\",stopped-threads=\"all\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Fails on FreeBSD apparently due to thread race conditions
    @skipIfLinux  # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_stopped_when_stopatentry_local(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped on entry (local)."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run with stop-at-entry flag
        self.runCmd("-interpreter-exec command \"process launch -s\"")
        self.expect("\^done")

        # Test that *stopped is printed
        # Note that message is different in Darwin and Linux:
        # Darwin: "*stopped,reason=\"signal-received\",signal-name=\"SIGSTOP\",signal-meaning=\"Stop\",frame={level=\"0\",addr=\"0x[0-9a-f]+\",func=\"_dyld_start\",file=\"??\",fullname=\"??\",line=\"-1\"},thread-id=\"1\",stopped-threads=\"all\"
        # Linux:
        # "*stopped,reason=\"end-stepping-range\",frame={addr=\"0x[0-9a-f]+\",func=\"??\",args=[],file=\"??\",fullname=\"??\",line=\"-1\"},thread-id=\"1\",stopped-threads=\"all\"
        self.expect(
            [
                "\*stopped,reason=\"signal-received\",signal-name=\"SIGSTOP\",signal-meaning=\"Stop\",frame=\{level=\"0\",addr=\"0x[0-9a-f]+\",func=\"_dyld_start\",file=\"\?\?\",fullname=\"\?\?\",line=\"-1\"\},thread-id=\"1\",stopped-threads=\"all\"",
                "\*stopped,reason=\"end-stepping-range\",frame={addr=\"0x[0-9a-f]+\",func=\"\?\?\",args=\[\],file=\"\?\?\",fullname=\"\?\?\",line=\"-1\"},thread-id=\"1\",stopped-threads=\"all\""])

        # Run to main to make sure we have not exited the application
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipUnlessDarwin
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_stopped_when_stopatentry_remote(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped on entry (remote)."""

        # Prepare debugserver
        import lldbgdbserverutils
        debugserver_exe = lldbgdbserverutils.get_debugserver_exe()
        if not debugserver_exe:
            self.skipTest("debugserver exe not found")
        hostname = "localhost"
        import random
        # the same as GdbRemoteTestCaseBase.get_next_port
        port = 12000 + random.randint(0, 3999)
        import pexpect
        debugserver_child = pexpect.spawn(
            "%s %s:%d" %
            (debugserver_exe, hostname, port))
        self.addTearDownHook(lambda: debugserver_child.terminate(force=True))

        self.spawnLldbMi(args=None)

        # Connect to debugserver
        self.runCmd(
            "-interpreter-exec command \"platform select remote-macosx --sysroot /\"")
        self.expect("\^done")
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.runCmd(
            "-interpreter-exec command \"process connect connect://%s:%d\"" %
            (hostname, port))
        self.expect("\^done")

        # Run with stop-at-entry flag
        self.runCmd("-interpreter-exec command \"process launch -s\"")
        self.expect("\^done")

        # Test that *stopped is printed
        self.expect(
            "\*stopped,reason=\"signal-received\",signal-name=\"SIGSTOP\",signal-meaning=\"Stop\",.+?thread-id=\"1\",stopped-threads=\"all\"")

        # Exit
        self.runCmd("-gdb-exit")
        self.expect("\^exit")

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux  # llvm.org/pr22841: lldb-mi tests fail on all Linux buildbots
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_stopped_when_segfault_local(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped when segfault occurred (local)."""

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

        # Set do_segfault=1 and run (to cause a segfault error)
        self.runCmd("-data-evaluate-expression \"do_segfault=1\"")
        self.expect("\^done,value=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")

        # Test that *stopped is printed
        # Note that message is different in Darwin and Linux:
        # Darwin: "*stopped,reason=\"exception-received\",exception=\"EXC_BAD_ACCESS (code=1, address=0x0)\",thread-id=\"1\",stopped-threads=\"all\""
        # Linux:  "*stopped,reason=\"exception-received\",exception=\"invalid
        # address (fault address:
        # 0x0)\",thread-id=\"1\",stopped-threads=\"all\""
        self.expect(["\*stopped,reason=\"exception-received\",exception=\"EXC_BAD_ACCESS \(code=1, address=0x0\)\",thread-id=\"1\",stopped-threads=\"all\"",
                     "\*stopped,reason=\"exception-received\",exception=\"invalid address \(fault address: 0x0\)\",thread-id=\"1\",stopped-threads=\"all\""])

    @skipUnlessDarwin
    def test_lldbmi_stopped_when_segfault_remote(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped when segfault occurred (remote)."""

        # Prepare debugserver
        import lldbgdbserverutils
        debugserver_exe = lldbgdbserverutils.get_debugserver_exe()
        if not debugserver_exe:
            self.skipTest("debugserver exe not found")
        hostname = "localhost"
        import random
        # the same as GdbRemoteTestCaseBase.get_next_port
        port = 12000 + random.randint(0, 3999)
        import pexpect
        debugserver_child = pexpect.spawn(
            "%s %s:%d" %
            (debugserver_exe, hostname, port))
        self.addTearDownHook(lambda: debugserver_child.terminate(force=True))

        self.spawnLldbMi(args=None)

        # Connect to debugserver
        self.runCmd(
            "-interpreter-exec command \"platform select remote-macosx --sysroot /\"")
        self.expect("\^done")
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.runCmd(
            "-interpreter-exec command \"process connect connect://%s:%d\"" %
            (hostname, port))
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        # FIXME -exec-run doesn't work
        # FIXME: self.runCmd("-exec-run")
        self.runCmd("-interpreter-exec command \"process launch\"")
        self.expect("\^done")  # FIXME: self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Set do_segfault=1 and run (to cause a segfault error)
        self.runCmd("-data-evaluate-expression \"do_segfault=1\"")
        self.expect("\^done,value=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")

        # Test that *stopped is printed
        self.expect(
            "\*stopped,reason=\"exception-received\",exception=\"EXC_BAD_ACCESS \(code=1, address=0x0\)\",thread-id=\"1\",stopped-threads=\"all\"")

        # Exit
        self.runCmd("-gdb-exit")
        self.expect("\^exit")
