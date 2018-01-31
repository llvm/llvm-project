"""
Test that breakpoints set on a bad address say they are bad.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class BadAddressBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_bad_address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        self.build()
        self.address_breakpoints()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        target, process, thread, bkpt = \
            lldbutil.run_to_source_breakpoint(self, 
                                              "Set a breakpoint here",
                                              lldb.SBFileSpec("main.c"))

        # Now see if we can read from 0.  If I can't do that, I don't
        # have a good way to know what an illegal address is...
        error = lldb.SBError()

        ptr = process.ReadPointerFromMemory(0x0, error)

        if not error.Success():
            bkpt = target.BreakpointCreateByAddress(0x0)
            for bp_loc in bkpt:
                self.assertTrue(bp_loc.IsResolved() == False)
        else:
            self.fail(
                "Could not find an illegal address at which to set a bad breakpoint.")
