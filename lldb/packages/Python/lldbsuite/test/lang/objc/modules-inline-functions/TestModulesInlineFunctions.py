"""Test that inline functions from modules are imported correctly"""

from __future__ import print_function



from distutils.version import StrictVersion

import unittest2
import os, time
import platform

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ModulesInlineFunctionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint here.')

    @skipUnlessDarwin
    @unittest2.skipIf(platform.system() != "Darwin" or StrictVersion('12.0.0') > platform.release(), "Only supported on Darwin 12.0.0+")
    @expectedFailureDarwin("llvm.org/pr25743")
    def test_expr(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("settings set target.clang-module-search-paths \"" + os.getcwd() + "\"")

        self.expect("expr @import myModule; 3", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["int", "3"])

        self.expect("expr isInline(2)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["4"])
