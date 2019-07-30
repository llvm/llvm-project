"""
Test completion for multiline expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class MultilineCompletionTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.c'

    def expect_string(self, string):
        import pexpect
        """This expects for "string", with timeout & EOF being test fails."""
        try:
            self.child.expect_exact(string)
        except pexpect.EOF:
            self.fail("Got EOF waiting for '%s'" % (string))
        except pexpect.TIMEOUT:
            self.fail("Timed out waiting for '%s'" % (string))

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_basic_completion(self):
        """Test that we can complete a simple multiline expression"""
        self.build()
        self.setTearDownCleanup()

        import pexpect
        exe = self.getBuildArtifact("a.out")
        prompt = "(lldb) "

        run_commands = ' -o "b main" -o "r"'
        self.child = pexpect.spawn(
            '%s %s %s %s' %
            (lldbtest_config.lldbExec, self.lldbOption, run_commands, exe))
        child = self.child

        self.expect_string(prompt)
        self.child.sendline("expr")
        self.expect_string("terminate with an empty line to evaluate")
        self.child.send("to_\t")
        self.expect_string("to_complete")

        self.deletePexpectChild()
