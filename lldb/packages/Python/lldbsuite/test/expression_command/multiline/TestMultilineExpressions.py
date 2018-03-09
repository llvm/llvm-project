"""Test multiline expressions."""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MultilineExpressionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number('main.c', 'break')

    @skipIfRemote
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_with_run_commands(self):
        """Test that multiline expressions work correctly"""
        self.build()
        import pexpect
        exe = self.getBuildArtifact("a.out")
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn(
            '%s %s %s' %
            (lldbtest_config.lldbExec, self.lldbOption, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Set the breakpoint, run the inferior, when it breaks, issue print on
        # the various convenience variables.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.c -l %d' % self.line)
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact("stop reason = breakpoint 1.1")
        child.expect_exact(prompt)
        child.sendline('expr')
        child.expect_exact('1:')

        child.sendline('2+')
        child.expect_exact('2:')

        child.sendline('3')
        child.expect_exact('3:')

        child.sendline('')
        child.expect_exact(prompt)
        self.expect(child.before, exe=False,
                    patterns=['= 5'])

    @skipIfRemote
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_empty_list(self):
        """Test printing an empty list of expressions"""
        import pexpect
        prompt = "(lldb) "

        # So that the child gets torn down after the test
        self.child = pexpect.spawn(
                "%s %s" %
                (lldbtest_config.lldbExec, self.lldbOption))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # We expect a prompt, then send "print" to start a list of expressions,
        # then an empty line. We expect a prompt back.
        child.expect_exact(prompt)
        child.sendline("print")
        child.expect_exact('1:')
        child.sendline("")
        child.expect_exact(prompt)
