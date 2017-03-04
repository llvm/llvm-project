"""
Test lldb-mi -environment-cd command.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiEnvironmentCdTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfDarwin   # Disabled while I investigate the failure on buildbot.
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_environment_cd(self):
        """Test that 'lldb-mi --interpreter' changes working directory for inferior."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # cd to a different directory
        self.runCmd("-environment-cd /tmp")
        self.expect("\^done")

        # Run to the end
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("@\"cwd: /tmp\\r\\n\"", exactly=True)
