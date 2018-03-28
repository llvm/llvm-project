"""
Test that we work properly with classes with the trivial_abi attribute
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators

class TestTrivialABI(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @decorators.skipUnlessSupportedTypeAttribute("trivial_abi")
    def test_call_trivial(self):
        """Test that we can print a variable & call a function with a trivial ABI class."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.expr_test(True)

    @decorators.skipUnlessSupportedTypeAttribute("trivial_abi")
    @decorators.expectedFailureAll(bugnumber="llvm.org/pr36870")
    def test_call_nontrivial(self):
        """Test that we can print a variable & call a function on the same class w/o the trivial ABI marker."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.expr_test(False)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def check_value(self, test_var, ivar_value):
        self.assertTrue(test_var.GetError().Success(), "Invalid valobj: %s"%(test_var.GetError().GetCString()))
        ivar = test_var.GetChildMemberWithName("ivar")
        self.assertTrue(test_var.GetError().Success(), "Failed to fetch ivar")
        self.assertEqual(ivar_value, ivar.GetValueAsSigned(), "Got the right value for ivar")
        
    def check_frame(self, thread):
        frame = thread.frames[0]
        inVal_var = frame.FindVariable("inVal")
        self.check_value(inVal_var, 10)

        options = lldb.SBExpressionOptions()
        inVal_expr = frame.EvaluateExpression("inVal", options)
        self.check_value(inVal_expr, 10)
        
        thread.StepOut()
        outVal_ret = thread.GetStopReturnValue()
        self.check_value(outVal_ret, 30)
        
    def expr_test(self, trivial):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file) 

        # Stop in a function that takes a trivial value, and try both frame var & expr to get its value:
        if trivial:
            self.check_frame(thread)
            return

        # Now continue to the same thing without the trivial_abi and see if we get that right:
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit my breakpoint the second time.")

        self.check_frame(threads[0])
        

