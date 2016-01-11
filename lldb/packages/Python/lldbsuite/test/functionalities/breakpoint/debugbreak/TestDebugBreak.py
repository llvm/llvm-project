"""
Test embedded breakpoints, like `asm int 3;` in x86 or or `__debugbreak` on Windows.
"""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class DebugBreakTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(archs=not_in(["i386", "i686"]))
    @no_debug_info_test
    def test_asm_int_3(self):
        """Test that intrinsics like `__debugbreak();` and `asm {"int3"}` are treated like breakpoints."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Run the program.
        target = self.dbg.CreateTarget(exe)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        # We've hit the first stop, so grab the frame.
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)

        # We should be in funciton 'bar'.
        self.assertTrue(frame.IsValid())
        function_name = frame.GetFunctionName()
        self.assertTrue('bar' in function_name)

        # We should be able to evaluate the parameter foo.
        value = frame.EvaluateExpression('*foo')
        self.assertEqual(value.GetValueAsSigned(), 42)

        # The counter should be 1 at the first stop and increase by 2 for each
        # subsequent stop.
        counter = 1
        while counter < 20:
          value = frame.EvaluateExpression('count')
          self.assertEqual(value.GetValueAsSigned(), counter)
          counter += 2
          process.Continue()

        # The inferior should exit after the last iteration.
        self.assertEqual(process.GetState(), lldb.eStateExited)
