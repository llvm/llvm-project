"""
Test that we obey thread conditioned breakpoints.
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ThreadSpecificBreakTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"])
    def test_python(self):
        """Test that we obey thread conditioned breakpoints."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # This test works by setting a breakpoint in a function conditioned to stop only on
        # the main thread, and then calling this function on a secondary thread, joining,
        # and then calling again on the main thread.  If the thread specific breakpoint works
        # then it should not be hit on the secondary thread, only on the main thread.

        main_source_spec = lldb.SBFileSpec ("main.cpp")

        main_breakpoint = target.BreakpointCreateBySourceRegex("Set main breakpoint here", main_source_spec);
        thread_breakpoint = target.BreakpointCreateBySourceRegex("Set thread-specific breakpoint here", main_source_spec)

        self.assertTrue(main_breakpoint.IsValid(), "Failed to set main breakpoint.")
        self.assertGreater(main_breakpoint.GetNumLocations(), 0, "main breakpoint has no locations associated with it.")
        self.assertTrue(thread_breakpoint.IsValid(), "Failed to set thread breakpoint.")
        self.assertGreater(thread_breakpoint.GetNumLocations(), 0, "thread breakpoint has no locations associated with it.")

        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(process, main_breakpoint)
        self.assertEqual(len(stopped_threads), 1, "main breakpoint stopped at unexpected number of threads")
        main_thread = stopped_threads[0]
        main_thread_id = main_thread.GetThreadID()

        # Set the thread-specific breakpoint to only stop on the main thread.  The run the function
        # on another thread and join on it.  If the thread-specific breakpoint works, the next
        # stop should be on the main thread.
        thread_breakpoint.SetThreadID(main_thread_id)

        process.Continue()
        next_stop_state = process.GetState()
        self.assertEqual(next_stop_state, lldb.eStateStopped, "We should have stopped at the thread breakpoint.")
        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(process, thread_breakpoint)
        self.assertEqual(len(stopped_threads), 1, "thread breakpoint stopped at unexpected number of threads")
        self.assertEqual(stopped_threads[0].GetThreadID(), main_thread_id, "thread breakpoint stopped at the wrong thread")
