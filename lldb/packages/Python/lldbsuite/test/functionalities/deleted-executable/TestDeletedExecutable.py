"""
Test process attach/resume.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestDeletedExecutable(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows # cannot delete a running executable
    @expectedFailureAll(oslist=["linux"]) # determining the architecture of the process fails
    @expectedFailureNetBSD
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)
        os.remove(exe)

        self.runCmd("process attach -p " + str(popen.pid))
        self.runCmd("kill")
