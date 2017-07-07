"""
Test expression command options.

Test cases:

o test_expr_options:
  Test expression command options.
"""

from __future__ import print_function


import os
import time
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class ExprOptionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.line = line_number('main.cpp', '// breakpoint_in_main')
        self.exe = os.path.join(os.getcwd(), "a.out")

    def test_expr_options(self):
        """These expression command options should work as expected."""
        self.build()

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, '// breakpoint_in_main', self.main_source_spec)

        frame = thread.GetFrameAtIndex(0)
        options = lldb.SBExpressionOptions()

        # test --language on C++ expression using the SB API's

        # Make sure we can evaluate a C++11 expression.
        val = frame.EvaluateExpression('foo != nullptr')
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.DebugSBValue(val)

        # Make sure it still works if language is set to C++11:
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus_11)
        val = frame.EvaluateExpression('foo != nullptr', options)
        self.assertTrue(val.IsValid())
        self.assertTrue(val.GetError().Success())
        self.DebugSBValue(val)

        # Make sure it fails if language is set to C:
        options.SetLanguage(lldb.eLanguageTypeC)
        val = frame.EvaluateExpression('foo != nullptr', options)
        self.assertTrue(val.IsValid())
        self.assertFalse(val.GetError().Success())
