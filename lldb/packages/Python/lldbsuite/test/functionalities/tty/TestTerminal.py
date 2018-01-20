"""
Test lldb command aliases.
"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LaunchInTerminalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Darwin is the only platform that I know of that supports optionally launching
    # a program in a separate terminal window. It would be great if other platforms
    # added support for this.
    @skipUnlessDarwin
    # If the test is being run under sudo, the spawned terminal won't retain that elevated
    # privilege so it can't open the socket to talk back to the test case
    @unittest2.skipIf(hasattr(os, 'geteuid') and os.geteuid()
                      == 0, "test cannot be run as root")
    # Do we need to disable this test if the testsuite is being run on a remote system?
    # This env var is only defined when the shell is running in a local mac
    # terminal window
    @unittest2.skipUnless(
        'TERM_PROGRAM' in os.environ,
        "test must be run on local system")
    @no_debug_info_test
    def test_launch_in_terminal(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        launch_info = lldb.SBLaunchInfo(["-lAF", "/tmp/"])
        launch_info.SetLaunchFlags(
            lldb.eLaunchFlagLaunchInTTY | lldb.eLaunchFlagCloseTTYOnExit)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        print("Error was: %s."%(error.GetCString()))
        self.assertTrue(
            error.Success(),
            "Make sure launch happened successfully in a terminal window")
        # Running in synchronous mode our process should have run and already
        # exited by the time target.Launch() returns
        self.assertTrue(process.GetState() == lldb.eStateExited)
