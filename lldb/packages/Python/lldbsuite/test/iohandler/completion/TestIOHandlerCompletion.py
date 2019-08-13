"""
Test completion in our IOHandlers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class IOHandlerCompletionTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

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
    def test_completion(self):
        self.setTearDownCleanup()

        import pexpect
        exe = self.getBuildArtifact("a.out")
        prompt = "(lldb) "

        self.child = pexpect.spawn(
            '%s %s %s %s' %
            (lldbtest_config.lldbExec, self.lldbOption, "", exe))

        self.expect_string(prompt)
        self.child.send("\t\t\t")
        self.expect_string("register")

        self.child.send("regi\t")
        self.expect_string(prompt + "register")
        self.child.send("\n")

        self.child.send("\t")
        self.expect_string("More (Y/n/a)")
        self.child.send("n")
        self.expect_string(prompt)

        # Shouldn't crash or anything like that.
        self.child.send("regoinvalid\t")
        self.expect_string(prompt)

        self.deletePexpectChild()
