"""
Make sure we can find the binary inside an app bundle.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import lldbsuite.test.lldbplatformutil as lldbplatformutil
from lldbsuite.test.lldbtest import *

@decorators.skipUnlessDarwin
class FindAppInMacOSAppBundle(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_find_app_in_bundle(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.find_app_in_bundle_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def find_app_in_bundle_test(self):
        """This reads in the .app, makes sure we get the right binary and can run it."""

        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        exe = self.getBuildArtifact("TestApp.app")
        error = lldb.SBError()
        target = self.dbg.CreateTarget(exe, None, None, False, error)
        self.assertTrue(error.Success(), "Could not create target: %s"%(error.GetCString()))
        self.assertTrue(target.IsValid(), "Target: TestApp.app is not valid.")
        exe_module_spec = target.GetExecutable()
        self.assertTrue(exe_module_spec.GetFilename(), "TestApp")

        bkpt = target.BreakpointCreateBySourceRegex("Set a breakpoint here", self.main_source_file)
        self.assertTrue(bkpt.GetNumLocations() == 1, "Couldn't set a breakpoint in the main app")

        if lldbplatformutil.getPlatform() == "macosx":
            launch_info = lldb.SBLaunchInfo(None)
            launch_info.SetWorkingDirectory(self.get_process_working_directory())

            error = lldb.SBError()
            process = target.Launch(launch_info, error)

            self.assertTrue(process.IsValid(), "Could not create a valid process for TestApp: %s"%(error.GetCString()))

            # Frame #0 should be at our breakpoint.
            threads = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)

            self.assertTrue(len(threads) == 1, "Expected 1 thread to stop at breakpoint, %d did."%(len(threads)))


