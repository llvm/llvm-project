"""
Test process attach.
"""

from __future__ import print_function


import os
import time
import lldb
import shutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

exe_name = "ProcessAttach"  # Must match Makefile


class ProcessAttachTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

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

    def test_attach_to_process_from_different_dir_by_id(self):
        """Test attach by process id"""
        try:
            os.mkdir(os.path.join(os.getcwd(),'newdir'))
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise
        testdir = os.getcwd()
        newdir = os.path.join(testdir,'newdir')
        exe = os.path.join(newdir, 'proc_attach')
        self.buildProgram('main.cpp', exe)
        self.addTearDownHook(lambda: shutil.rmtree(newdir))

        # Spawn a new process
        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        os.chdir('newdir')
        self.addTearDownHook(lambda: os.chdir(testdir))
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
