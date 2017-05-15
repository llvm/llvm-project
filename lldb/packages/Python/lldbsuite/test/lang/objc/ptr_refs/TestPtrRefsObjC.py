"""
Test the ptr_refs tool on Darwin with Objective-C
"""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrRefsObjC(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_ptr_refs(self):
        """Test the ptr_refs tool on Darwin with Objective-C"""
        self.build()
        exe_name = 'a.out'
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        main_file_spec = lldb.SBFileSpec('main.m')
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break', main_file_spec)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1 and the break condition should hold.
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")

        frame = thread.GetFrameAtIndex(0)

        self.dbg.HandleCommand("script import lldb.macosx.heap")
        self.expect("ptr_refs self", substrs=["malloc", "stack"])

