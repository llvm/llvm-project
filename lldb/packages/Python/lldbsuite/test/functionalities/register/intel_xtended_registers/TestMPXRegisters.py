"""
Test the MPX registers.
"""

from __future__ import print_function


import os
import sys
import time
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

    @skipIf(compiler="clang")
    @skipIf(oslist=["linux"], compiler="gcc", compiler_version=["<", "5"]) #GCC version >= 5 supports MPX.
    @skipIf(oslist=no_match(['linux']))
    @skipIf(archs=no_match(['i386', 'x86_64']))
    def test_mpx_registers_with_example_code(self):
        """Test MPX registers with example code."""
        self.build()
        self.mpx_registers_with_example_code()

    def mpx_registers_with_example_code(self):
        """Test MPX registers after running example code."""
        self.line = line_number('main.cpp', '// Set a break point here.')

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        if (process.GetState() == lldb.eStateExited):
            self.skipTest("HW doesn't support MPX feature.")
        else:
            self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                        substrs = ["stop reason = breakpoint 1."])

        if self.getArchitecture() == 'x86_64':
            self.expect("register read -s 3",
                        substrs = ['bnd0 = {0x0000000000000010 0xffffffffffffffe6}',
                                   'bnd1 = {0x0000000000000020 0xffffffffffffffd6}',
                                   'bnd2 = {0x0000000000000030 0xffffffffffffffc6}',
                                   'bnd3 = {0x0000000000000040 0xffffffffffffffb6}',
                                   'bndcfgu = {0x01 0x80 0xb5 0x76 0xff 0x7f 0x00 0x00}',
                                   'bndstatus = {0x02 0x80 0xb5 0x76 0xff 0x7f 0x00 0x00}'])
        if self.getArchitecture() == 'i386':
            self.expect("register read -s 3",
                        substrs = ['bnd0 = {0x0000000000000010 0x00000000ffffffe6}',
                                   'bnd1 = {0x0000000000000020 0x00000000ffffffd6}',
                                   'bnd2 = {0x0000000000000030 0x00000000ffffffc6}',
                                   'bnd3 = {0x0000000000000040 0x00000000ffffffb6}',
                                   'bndcfgu = {0x01 0xd0 0x7d 0xf7 0x00 0x00 0x00 0x00}',
                                   'bndstatus = {0x02 0xd0 0x7d 0xf7 0x00 0x00 0x00 0x00}'])

