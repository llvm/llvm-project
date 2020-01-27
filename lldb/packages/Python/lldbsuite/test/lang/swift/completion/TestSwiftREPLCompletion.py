
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class SwiftCompletionTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipUnlessDarwin
    def test_basic_completion(self):

        self.launch(extra_args=["--repl"], executable=None, dimensions=(100,500))

        # Wait on the first prompt
        self.child.expect_exact("1>")
        # Press tab a few times which should do nothing.
        # Note that we don't get any indentation whitespace as
        # pexpect is not recognized as a interactive terminal by pexpect it seems.
        self.child.send("\t\t\t")

        # Try completing something that only has one result "Hasabl" -> "Hashable".
        self.child.send("Hashabl\t")
        self.child.expect_exact("Hashable")
        self.child.sendline("")

        # Try completing something that has multiple completions.
        self.child.send("Hash\t")
        self.child.expect_exact("Available completions:")
        self.child.expect_exact("Hashable")
        self.child.expect_exact("Hasher")
        self.child.sendline("")

    def setUpCommands(self):
        return [] # REPL doesn't take any setup commands.

    def expect_prompt(self):
        pass # No constant prompt on the REPL.
