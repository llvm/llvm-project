"""
Test completion in our IOHandlers.
"""

import os

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
            (lldbtest_config.lldbExec, self.lldbOption, "", exe),
            dimensions=(100, 500))

        self.expect_string(prompt)
        # Start tab completion, go to the next page and then display all with 'a'.
        self.child.send("\t\ta")
        self.expect_string("register")

        # Try tab completing regi to register.
        self.child.send("regi\t")
        self.expect_string(prompt + "register")
        self.child.send("\n")

        # Try tab completing directories and files. Also tests the partial
        # completion where LLDB shouldn't print a space after the directory
        # completion (as it didn't completed the full token).
        dir_without_slashes = os.path.realpath(os.path.dirname(__file__)).rstrip("/")
        self.child.send("file " + dir_without_slashes + "\t")
        self.expect_string("iohandler/completion/")
        # If we get a correct partial completion without a trailing space, then this
        # should complete the current test file.
        self.child.send("TestIOHandler\t")
        self.expect_string("TestIOHandlerCompletion.py")
        self.child.send("\n")

        # Start tab completion and abort showing more commands with 'n'.
        self.child.send("\t")
        self.expect_string("More (Y/n/a)")
        self.child.send("n")
        self.expect_string(prompt)

        # Shouldn't crash or anything like that.
        self.child.send("regoinvalid\t")
        self.expect_string(prompt)

        self.deletePexpectChild()
