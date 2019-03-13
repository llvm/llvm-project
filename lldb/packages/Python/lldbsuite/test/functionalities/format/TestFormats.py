"""
Test the command history mechanism
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFormats(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        hostoslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_formats(self):
        """Test format string functionality."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        import pexpect
        prompt = "(lldb) "
        child = pexpect.spawn(
            '%s %s -x -o "b main" -o r %s' %
            (lldbtest_config.lldbExec, self.lldbOption, exe))
        # So that the spawned lldb session gets shutdown durng teardown.
        self.child = child

        # Substitute 'Help!' for 'help' using the 'commands regex' mechanism.
        child.expect_exact(prompt + 'target create "%s"' % exe)
        child.expect_exact(prompt + 'b main')
        child.expect_exact(prompt + 'r')
        child.expect_exact(prompt)
        child.sendline()
