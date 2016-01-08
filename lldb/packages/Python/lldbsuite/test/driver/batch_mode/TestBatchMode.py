"""
Test that the lldb driver's batch mode works correctly.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *

class DriverBatchModeTest (TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        self.line = line_number(self.source, 'Stop here to unset keep_waiting')
        self.victim = None

    def expect_string (self, string):
        import pexpect
        """This expects for "string", with timeout & EOF being test fails."""
        try:
            self.child.expect_exact(string)
        except pexpect.EOF:
            self.fail ("Got EOF waiting for '%s'"%(string))
        except pexpect.TIMEOUT:
            self.fail ("Timed out waiting for '%s'"%(string))

    @skipIfRemote # test not remote-ready llvm.org/pr24813
    @expectedFlakeyFreeBSD("llvm.org/pr25172 fails rarely on the buildbot")
    @expectedFlakeyLinux("llvm.org/pr25172")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_batch_mode_run_crash (self):
        """Test that the lldb driver's batch mode works correctly."""
        self.build()
        self.setTearDownCleanup()

        import pexpect
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "

        # Pass CRASH so the process will crash and stop in batch mode.
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


    @skipIfRemote # test not remote-ready llvm.org/pr24813
    @expectedFlakeyFreeBSD("llvm.org/pr25172 fails rarely on the buildbot")
    @expectedFlakeyLinux("llvm.org/pr25172")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_batch_mode_run_exit (self):
        """Test that the lldb driver's batch mode works correctly."""
        self.build()
        self.setTearDownCleanup()

        import pexpect
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "

        # Now do it again, and make sure if we don't crash, we quit:
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

    def closeVictim(self):
        if self.victim != None:
            self.victim.close()
            self.victim = None

    @skipIfRemote # test not remote-ready llvm.org/pr24813
    @expectedFlakeyFreeBSD("llvm.org/pr25172 fails rarely on the buildbot")
    @expectedFlakeyLinux("llvm.org/pr25172")
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_batch_mode_attach_exit (self):
        """Test that the lldb driver's batch mode works correctly."""
        self.build()
        self.setTearDownCleanup()
 
        import pexpect
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "

        # Finally, start up the process by hand, attach to it, and wait for its completion.
        # Attach is funny, since it looks like it stops with a signal on most Unixen so 
        # care must be taken not to treat that as a reason to exit batch mode.
        
        # Start up the process by hand and wait for it to get to the wait loop.

        self.victim = pexpect.spawn('%s WAIT' %(exe))
        if self.victim == None:
            self.fail("Could not spawn ", exe, ".")

        self.addTearDownHook (self.closeVictim)

        if self.TraceOn():
            self.victim.logfile_read = sys.stdout

        self.victim.expect("PID: ([0-9]+) END")
        if self.victim.match == None:
            self.fail("Couldn't get the target PID.")

        victim_pid = int(self.victim.match.group(1))
        
        self.victim.expect("Waiting")

        run_commands = ' -b -o "process attach -p %d" -o "breakpoint set --file %s --line %d -N keep_waiting" -o "continue" -o "break delete keep_waiting" -o "expr keep_waiting = 0" -o "continue" ' % (victim_pid, self.source, self.line)
        self.child = pexpect.spawn('%s %s %s %s' % (lldbtest_config.lldbExec, self.lldbOption, run_commands, exe))

        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We should see the "run":
        self.expect_string ("attach")

        self.expect_string(prompt + "continue")

        self.expect_string(prompt + "continue")

        # Then we should see the process exit:
        self.expect_string ("Process %d exited with status"%(victim_pid))
        
        victim_index = self.victim.expect([pexpect.EOF, pexpect.TIMEOUT])
        self.assertTrue(victim_index == 0, "Victim didn't really exit.")

        index = self.child.expect([pexpect.EOF, pexpect.TIMEOUT])
        self.assertTrue(index == 0, "lldb didn't close on successful batch completion.")

        
