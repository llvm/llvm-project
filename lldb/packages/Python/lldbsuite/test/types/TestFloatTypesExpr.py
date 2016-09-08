"""
Test that variable expressions of floating point types are evaluated correctly.
"""

from __future__ import print_function


import AbstractBase
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FloatTypesExprTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    # rdar://problem/8493023
    # test/types failures for Test*TypesExpr.py: element offset computed wrong
    # and sign error?

    def setUp(self):
        # Call super's setUp().
        AbstractBase.GenericTester.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def test_float_type(self):
        """Test that float-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('float.cpp', set(['float']))

    @skipUnlessDarwin
    def test_float_type_from_block(self):
        """Test that float-type variables are displayed correctly from a block."""
        self.build_and_run_expr('float.cpp', set(['float']), bc=True)

    def test_double_type(self):
        """Test that double-type variable expressions are evaluated correctly."""
        self.build_and_run_expr('double.cpp', set(['double']))

    @skipUnlessDarwin
    def test_double_type_from_block(self):
        """Test that double-type variables are displayed correctly from a block."""
        self.build_and_run_expr('double.cpp', set(['double']), bc=True)
