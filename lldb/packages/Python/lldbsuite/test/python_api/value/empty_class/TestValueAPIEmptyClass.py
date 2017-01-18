from __future__ import print_function

import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ValueAPIEmptyClassTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test(self):
        self.build()
        exe = os.path.join(os.getcwd(), 'a.out')
        line = line_number('main.cpp', '// Break at this line')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation('main.cpp', line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Verify that we can access to a frame variable with an empty class type
        e = frame0.FindVariable('e')
        self.assertTrue(e.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(e)
        self.assertEqual(e.GetNumChildren(), 0)

        # Verify that we can acces to a frame variable what is a pointer to an
        # empty class
        ep = frame0.FindVariable('ep')
        self.assertTrue(ep.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(ep)

        # Verify that we can dereference a pointer to an empty class
        epd = ep.Dereference()
        self.assertTrue(epd.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(epd)
        self.assertEqual(epd.GetNumChildren(), 0)

