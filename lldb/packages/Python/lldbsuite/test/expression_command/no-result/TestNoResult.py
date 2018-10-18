"""
Test that an expression that returns no result returns a sensible error.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestExprNoResult(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_no_result(self):
        """Run an expression that has no result, check the error."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.sample_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def sample_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        result = frame.EvaluateExpression("int $x = 10")
        # No result expressions are considered to fail:
        self.assertTrue(result.GetError().Fail(), "An expression with no result is a failure.")
        # But the reason should be eExpressionProducedNoResult
        self.assertEqual(result.GetError().GetError(), lldb.eExpressionProducedNoResult, 
                         "But the right kind of failure")
