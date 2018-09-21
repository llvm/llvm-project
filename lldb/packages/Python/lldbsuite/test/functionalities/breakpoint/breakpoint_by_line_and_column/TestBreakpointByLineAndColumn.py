"""
Test setting a breakpoint by line and column.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointByLineAndColumnTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLineAndColumn(self):
        self.build()
        main_c = lldb.SBFileSpec("main.c")
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self,
                                                              main_c, 20, 50)
        self.expect("fr v did_call", substrs='1')
        in_then = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), 20)
            in_then |= b_loc.GetColumn() == 50
        self.assertTrue(in_then)

    ## Skip gcc version less 7.1 since it doesn't support -gcolumn-info
    @skipIf(compiler="gcc", compiler_version=['<', '7.1'])
    def testBreakpointByLine(self):
        self.build()
        main_c = lldb.SBFileSpec("main.c")
        _, _, _, breakpoint = lldbutil.run_to_line_breakpoint(self, main_c, 20)
        self.expect("fr v did_call", substrs='0')
        in_condition = False
        for i in range(breakpoint.GetNumLocations()):
            b_loc = breakpoint.GetLocationAtIndex(i).GetAddress().GetLineEntry()
            self.assertEqual(b_loc.GetLine(), 20)
            in_condition |= b_loc.GetColumn() < 30
        self.assertTrue(in_condition)
