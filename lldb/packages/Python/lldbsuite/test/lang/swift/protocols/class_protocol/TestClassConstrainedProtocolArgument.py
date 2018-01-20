"""
Test that variables passed in as a class constrained protocol type
are correctly printed.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators

class TestClassConstrainedProtocolArgument(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @decorators.expectedFailureAll("https://bugs.swift.org/browse/SR-6657")
    def test_class_constrained_protocol(self):
        """Test that class constrained protocol types are correctly printed."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.swift")
        self.do_class_constrained()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_class_constrained(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Break here and print input", self.main_source_file) 

        frame = thread.GetFrameAtIndex(0)
        input_var = frame.FindVariable("input")
        self.assertTrue(input_var.GetError().Success(), "Failed to fetch test_var")
        input_typename = input_var.GetTypeName()
        self.assertTrue(input_typename != "T", "Couldn't get the real type.")
        

