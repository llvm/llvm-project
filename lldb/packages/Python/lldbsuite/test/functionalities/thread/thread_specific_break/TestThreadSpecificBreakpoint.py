"""
Test that we obey thread conditioned breakpoints.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

def set_thread_id(thread, breakpoint):
    id = thread.id
    breakpoint.SetThreadID(id)

def set_thread_name(thread, breakpoint):
    breakpoint.SetThreadName("main-thread")

class ThreadSpecificBreakTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(['pyapi'])

    @expectedFailureAll(oslist=["windows"])
    @expectedFailureAll(oslist=['ios', 'watchos', 'tvos', 'bridgeos'], archs=['armv7', 'armv7k'], bugnumber='rdar://problem/34563920') # armv7 ios problem - breakpoint with tid qualifier isn't working
    def test_thread_id(self):
        self.do_test(set_thread_id)

    @skipUnlessDarwin
    @expectedFailureAll(oslist=['ios', 'watchos', 'tvos', 'bridgeos'], archs=['armv7', 'armv7k'], bugnumber='rdar://problem/34563920') # armv7 ios problem - breakpoint with tid qualifier isn't working
    def test_thread_name(self):
        self.do_test(set_thread_name)

    def do_test(self, setter_method):
        """Test that we obey thread conditioned breakpoints."""
        self.build()
        main_source_spec = lldb.SBFileSpec("main.cpp")
        (target, process, main_thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
                "Set main breakpoint here", main_source_spec)

        main_thread_id = main_thread.GetThreadID()

        # This test works by setting a breakpoint in a function conditioned to stop only on
        # the main thread, and then calling this function on a secondary thread, joining,
        # and then calling again on the main thread.  If the thread specific breakpoint works
        # then it should not be hit on the secondary thread, only on the main
        # thread.
        thread_breakpoint = target.BreakpointCreateBySourceRegex(
            "Set thread-specific breakpoint here", main_source_spec)
        self.assertGreater(
            thread_breakpoint.GetNumLocations(),
            0,
            "thread breakpoint has no locations associated with it.")

        # Set the thread-specific breakpoint to only stop on the main thread.  The run the function
        # on another thread and join on it.  If the thread-specific breakpoint works, the next
        # stop should be on the main thread.

        main_thread_id = main_thread.GetThreadID()
        setter_method(main_thread, thread_breakpoint)

        process.Continue()
        next_stop_state = process.GetState()
        self.assertEqual(
            next_stop_state,
            lldb.eStateStopped,
            "We should have stopped at the thread breakpoint.")
        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, thread_breakpoint)
        self.assertEqual(
            len(stopped_threads),
            1,
            "thread breakpoint stopped at unexpected number of threads")
        self.assertEqual(
            stopped_threads[0].GetThreadID(),
            main_thread_id,
            "thread breakpoint stopped at the wrong thread")
