"""
Test that the lldb-mi driver understands MI command syntax.
"""

from __future__ import print_function



import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class MiSyntaxTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_tokens(self):
        """Test that 'lldb-mi --interpreter' prints command tokens."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("000-file-exec-and-symbols %s" % self.myexe)
        self.expect("000\^done")

        # Run to main
        self.runCmd("100000001-break-insert -f main")
        self.expect("100000001\^done,bkpt={number=\"1\"")
        self.runCmd("2-exec-run")
        self.expect("2\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Exit
        self.runCmd("0000000000000000000003-exec-continue")
        self.expect("0000000000000000000003\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_specialchars(self):
        """Test that 'lldb-mi --interpreter' handles complicated strings."""

        # Create an alias for myexe
        complicated_myexe = "C--mpl-x file's`s @#$%^&*()_+-={}[]| name"
        os.symlink(self.myexe, complicated_myexe)
        self.addTearDownHook(lambda: os.unlink(complicated_myexe))

        self.spawnLldbMi(args = "\"%s\"" % complicated_myexe)

        # Test that the executable was loaded
        self.expect("-file-exec-and-symbols \"%s\"" % complicated_myexe, exactly = True)
        self.expect("\^done")

        # Check that it was loaded correctly
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @expectedFailureLinux  # Failing in ~6/600 dosep runs (build 3120-3122)
    def test_lldbmi_process_output(self):
        """Test that 'lldb-mi --interpreter' wraps process output correctly."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")

        # Test that a process output is wrapped correctly
        self.expect("\@\"'\\\\r\\\\n\"")
        self.expect("\@\"` - it's \\\\\\\\n\\\\x12\\\\\"\\\\\\\\\\\\\"")
