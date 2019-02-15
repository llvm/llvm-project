"""Test that importing modules in C++ works as expected."""

from __future__ import print_function


from distutils.version import StrictVersion
import unittest2
import os
import time
import lldb
import platform

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CXXModulesImportTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "10.12"])
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.cpp'))

        self.expect("expr -l Objective-C++ -- @import Bar")
        self.expect("expr -- Bar()", substrs = ["success"])
