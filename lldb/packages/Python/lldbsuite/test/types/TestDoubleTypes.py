"""
Test that variables of floating point types are displayed correctly.
"""

from __future__ import print_function


import AbstractBase
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DoubleTypesTestCase(AbstractBase.GenericTester):

    mydir = AbstractBase.GenericTester.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        AbstractBase.GenericTester.setUp(self)
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def test_double_type(self):
        """Test that double-type variables are displayed correctly."""
        self.build_and_run('double.cpp', set(['double']))

    @skipUnlessDarwin
    def test_double_type_from_block(self):
        """Test that double-type variables are displayed correctly from a block."""
        self.build_and_run('double.cpp', set(['double']), bc=True)
