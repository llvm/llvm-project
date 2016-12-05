"""Test the integrity of the lldb public api directory containing SB*.h headers.

There should be nothing unwanted there and a simpe main.cpp which includes SB*.h
should compile and link with the LLDB framework."""

from __future__ import print_function


import os
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBDirCheckerCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.cpp'
        self.exe_name = 'a.out'
        self.generateSource(self.source)

    @skipIfNoSBHeaders
    def test_sb_api_directory(self):
        """Test the SB API directory and make sure there's no unwanted stuff."""

        # Only proceed if this is an Apple OS, "x86_64", and local platform.
        if not (self.platformIsDarwin() and self.getArchitecture() == "x86_64"):
            self.skipTest("This test is only for LLDB.framework built 64-bit")
        if self.getArchitecture() == "i386":
            self.skipTest(
                "LLDB is 64-bit and cannot be linked to 32-bit test program.")

        self.buildDriver(self.source, self.exe_name)
        self.sanity_check_executable(self.exe_name)

    def sanity_check_executable(self, exe_name):
        """Sanity check executable compiled from the auto-generated program."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file %s" % exe, CURRENT_EXECUTABLE_SET)

        self.line_to_break = line_number(
            self.source, '// Set breakpoint here.')

        env_cmd = "settings set target.env-vars %s=%s" % (
            self.dylibPath, self.getLLDBLibraryEnvVal())
        if self.TraceOn():
            print("Set environment to: ", env_cmd)
        self.runCmd(env_cmd)
        self.addTearDownHook(
            lambda: self.dbg.HandleCommand(
                "settings remove target.env-vars %s" %
                self.dylibPath))

        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line_to_break, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.runCmd('frame variable')
