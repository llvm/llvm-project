"""
Test the IR interpreter
"""

from __future__ import print_function

import unittest2

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class IRInterpreterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c',
                                '// Set breakpoint here')

        # Disable confirmation prompt to avoid infinite wait
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def build_and_run(self):
        """Test the IR interpreter"""
        self.build()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

    @add_test_categories(['pyapi'])
    # getpid() is POSIX, among other problems, see bug
    @expectedFailureAll(
        oslist=['windows'],
        bugnumber="http://llvm.org/pr21765")
    @expectedFailureAll(
        oslist=['linux'],
        archs=['arm'],
        bugnumber="llvm.org/pr27868")
    def test_ir_interpreter(self):
        self.build_and_run()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)

        set_up_expressions = ["int $i = 9", "int $j = 3", "int $k = 5"]

        expressions = ["$i + $j",
                       "$i - $j",
                       "$i * $j",
                       "$i / $j",
                       "$i % $k",
                       "$i << $j",
                       "$i & $j",
                       "$i | $j",
                       "$i ^ $j"]

        for expression in set_up_expressions:
            self.frame().EvaluateExpression(expression, options)

        for expression in expressions:
            interp_expression = expression
            jit_expression = "(int)getpid(); " + expression

            interp_result = self.frame().EvaluateExpression(
                interp_expression, options).GetValueAsSigned()
            jit_result = self.frame().EvaluateExpression(
                jit_expression, options).GetValueAsSigned()

            self.assertEqual(
                interp_result,
                jit_result,
                "While evaluating " +
                expression)
