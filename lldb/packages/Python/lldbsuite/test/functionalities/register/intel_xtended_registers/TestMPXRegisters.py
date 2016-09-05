"""
Test the MPX registers.
"""

from __future__ import print_function

import os, sys, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.has_teardown = False

    def tearDown(self):
        self.dbg.GetSelectedTarget().GetProcess().Destroy()
        TestBase.tearDown(self)

    @skipIfiOSSimulator
    @skipIf(compiler="clang")
    @expectedFailureAll(oslist=["linux"], compiler="gcc", compiler_version=["<", "5"])
    @skipIf(archs=no_match(['amd64', 'i386', 'x86_64']))
    def test_mpx_registers_with_example_code(self):
        """Test MPX registers with example code."""
        self.build()
        self.mpx_registers_with_example_code()

    def mpx_registers_with_example_code(self):
        """Test MPX registers after running example code."""
        self.line = line_number('main.cpp', '// Set a break point here.')

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd('settings set target.inline-breakpoint-strategy always')
        self.addTearDownHook(
            lambda: self.runCmd("settings set target.inline-breakpoint-strategy always"))

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs = ["stop reason = breakpoint 1."])

        self.expect("register read -s 3",
                    substrs = ['bnd0 = {0x0000000000000010 0xffffffffffffffe6}',
                               'bnd1 = {0x0000000000000020 0xffffffffffffffd6}',
                               'bnd2 = {0x0000000000000030 0xffffffffffffffc6}',
                               'bnd3 = {0x0000000000000040 0xffffffffffffffb6}',
                               'bndcfgu = {0x01 0x80 0xb5 0x76 0xff 0x7f 0x00 0x00}',
                               'bndstatus = {0x02 0x80 0xb5 0x76 0xff 0x7f 0x00 0x00}'])
