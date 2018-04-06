"""
Make sure we don't trap exceptions with SetREPLEnabled(true)
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestREPLExceptions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the 
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfLinux # <rdar://problem/39245862>
    def test_repl_exceptions(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.swift")
        self.do_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file) 

        frame = thread.GetFrameAtIndex(0)
        options = lldb.SBExpressionOptions()
        options.SetREPLMode(True)
        val = frame.EvaluateExpression("f_with_exceptions(); 5", options)
        self.assertTrue(val.GetError().Success(), 
                        "Got an error evaluating expression: %s."%(val.GetError().GetCString()))
        self.assertEqual(val.GetValueAsUnsigned(), 5,"The expression didn't return the correct result")
