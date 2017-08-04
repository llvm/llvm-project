"""
Test lldb-mi -thread-info command.
"""

from __future__ import print_function

import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiThreadInfoTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # pthreads not supported on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_thread_info(self):
        """Test that -thread-info prints thread info and the current-thread-id"""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        self.runCmd("-break-insert ThreadProc")
        self.expect("\^done")

        # Run to the breakpoint
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        self.runCmd("-thread-info")
        self.expect(
            "\^done,threads=\[\{id=\"1\",(.*)\},\{id=\"2\",(.*)\],current-thread-id=\"2\"")

        self.runCmd("-gdb-quit")
