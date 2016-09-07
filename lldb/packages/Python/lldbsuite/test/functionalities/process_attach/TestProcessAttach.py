"""
Test process attach.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

exe_name = "ProcessAttach"  # Must match Makefile


class ProcessAttachTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfiOSSimulator
    def test_attach_to_process_by_id(self):
        """Test attach by process id"""
        self.build()
        exe = os.path.join(os.getcwd(), exe_name)

        # Spawn a new process
        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

    def test_attach_to_process_by_name(self):
        """Test attach by process name"""
        self.build()
        exe = os.path.join(os.getcwd(), exe_name)

        # Spawn a new process
        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -n " + exe_name)

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

    def tearDown(self):
        # Destroy process before TestBase.tearDown()
        self.dbg.GetSelectedTarget().GetProcess().Destroy()

        # Call super's tearDown().
        TestBase.tearDown(self)
