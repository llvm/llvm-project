"""
Test that the lldb driver's batch mode works correctly.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *

class DriverBatchModeTest (TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote # test not remote-ready llvm.org/pr24813
    @expectedFlakeyFreeBSD("llvm.org/pr25172 fails rarely on the buildbot")
    @expectedFlakeyLinux("llvm.org/pr25172")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_driver_batch_mode(self):
        """Test that the lldb driver's batch mode works correctly."""
        self.build()
        self.setTearDownCleanup()
        self.batch_mode()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'

    def expect_string (self, string):
        import pexpect
        """This expects for "string", with timeout & EOF being test fails."""
        try:
            self.child.expect_exact(string)
        except pexpect.EOF:
            self.fail ("Got EOF waiting for '%s'"%(string))
        except pexpect.TIMEOUT:
            self.fail ("Timed out waiting for '%s'"%(string))

    def batch_mode (self):
        import pexpect
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "

        # First time through, pass CRASH so the process will crash and stop in batch mode.
        run_commands = ' -b -o "break set -n main" -o "run" -o "continue" -k "frame var touch_me_not"'
        self.child = pexpect.spawn('%s %s %s %s -- CRASH' % (lldbtest_config.lldbExec, self.lldbOption, run_commands, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We should see the "run":
        self.expect_string ("run")
        # We should have hit the breakpoint & continued:
        self.expect_string ("continue")
        # The App should have crashed:
        self.expect_string("About to crash")
        # The -k option should have printed the frame variable once:
        self.expect_string ('(char *) touch_me_not')
        # Then we should have a live prompt:
        self.expect_string (prompt)
        self.child.sendline("frame variable touch_me_not")
        self.expect_string ('(char *) touch_me_not')
        
        self.deletePexpectChild()

        # Now do it again, and see make sure if we don't crash, we quit:
        run_commands = ' -b -o "break set -n main" -o "run" -o "continue" '
        self.child = pexpect.spawn('%s %s %s %s -- NOCRASH' % (lldbtest_config.lldbExec, self.lldbOption, run_commands, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We should see the "run":
        self.expect_string ("run")
        # We should have hit the breakpoint & continued:
        self.expect_string ("continue")
        # The App should have not have crashed:
        self.expect_string("Got there on time and it did not crash.")
        # Then we should have a live prompt:
        self.expect_string ("exited")
        index = self.child.expect([pexpect.EOF, pexpect.TIMEOUT])
        self.assertTrue(index == 0, "lldb didn't close on successful batch completion.")

