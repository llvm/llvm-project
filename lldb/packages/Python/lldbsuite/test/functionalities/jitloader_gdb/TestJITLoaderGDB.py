"""Test for the JITLoaderGDB interface"""

from __future__ import print_function



import unittest2
import os
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
import re


class JITLoaderGDBTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipTestIfFn(lambda x: (True, "Skipped because the test crashes the test runner"), bugnumber="llvm.org/pr24702")
    @unittest2.expectedFailure("llvm.org/pr24702")
    def test_bogus_values(self):
        """Test that we handle inferior misusing the GDB JIT interface"""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # launch the process, do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The inferior will now pass bogus values over the interface. Make sure we don't crash.

        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
