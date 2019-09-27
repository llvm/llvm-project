"""
Test Write Register
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_threads


class WriteRegisterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_write_register(self):
        """Use Python APIs to check write of DPU register."""
        self.build()
        self.do_test_write_register()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test_write_register(self):
        exe = self.getBuildArtifact("a.out")
        filespec = lldb.SBFileSpec("main.c", False)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByName("main")
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        self.runCmd("register write r0 0xaa")

        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateExited)
        self.assertTrue(process.GetExitStatus() == 0xab)
