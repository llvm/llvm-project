"""Test that importing modules in Objective-C works as expected."""

from __future__ import print_function



import unittest2
import os, time
import platform
from distutils.version import StrictVersion

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ObjCModulesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    @skipUnlessDarwin
    @unittest2.expectedFailure("rdar://20416388")
    @unittest2.skipIf(platform.system() != "Darwin" or StrictVersion('12.0.0') > platform.release(), "Only supported on Darwin 12.0.0+")
    def test_expr(self):
        if not self.applies():
            return

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

        self.expect("expr @import Darwin; 3", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["int", "3"])

        self.expect("expr getpid()", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["pid_t"])

        self.expect("expr @import Foundation; 4", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["int", "4"])

        self.expect("expr string.length", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["NSUInteger", "5"])

        self.expect("expr array.count", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["NSUInteger", "3"])

        self.expect("p *[NSURL URLWithString:@\"http://lldb.llvm.org\"]", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["NSURL", "isa", "_urlString"])

        self.expect("p [NSURL URLWithString:@\"http://lldb.llvm.org\"].scheme", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["http"])
