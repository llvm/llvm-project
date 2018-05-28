"""
Test the error message if the SDK the program was built against doesn't exist.
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestMissingSDK(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @skipIf(oslist=['linux', 'windows'])
    def testMissingSDK(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec("main.swift"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)
        process = target.LaunchSimple(None, None, os.getcwd())
        self.expect("p message", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs = ["Hello"])

