"""
Test some lldb apropos commands.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AproposCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_apropos_variable(self):
        """Test that 'apropos variable' prints the fully qualified command name"""
        self.expect(
            'apropos variable',
            substrs=[
                'frame variable',
                'target variable',
                'watchpoint set variable'])
