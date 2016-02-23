"""
Testlldb Python SBFrame APIs IsInlined() and GetFunctionName().
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class InlinedFrameAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.source = 'inlines.c'
        self.first_stop = line_number(self.source, '// This should correspond to the first break stop.')
        self.second_stop = line_number(self.source, '// This should correspond to the second break stop.')

    @add_test_categories(['pyapi'])
    def test_stop_at_outer_inline(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by the name of 'inner_inline'.
        breakpoint = target.BreakpointCreateByName('inner_inline', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() > 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        import lldbsuite.test.lldbutil as lldbutil
        stack_traces1 = lldbutil.print_stacktraces(process, string_buffer=True)
        if self.TraceOn():
            print("Full stack traces when first stopped on the breakpoint 'inner_inline':")
            print(stack_traces1)

        # The first breakpoint should correspond to an inlined call frame.
        # If it's an inlined call frame, expect to find, in the stack trace,
        # that there is a frame which corresponds to the following call site:
        #
        #     outer_inline (argc);
        #
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frame0 = thread.GetFrameAtIndex(0)
        if frame0.IsInlined():
            filename = frame0.GetLineEntry().GetFileSpec().GetFilename()
            self.assertTrue(filename == self.source)
            self.expect(stack_traces1, "First stop at %s:%d" % (self.source, self.first_stop), exe=False,
                        substrs = ['%s:%d' % (self.source, self.first_stop)])

            # Expect to break again for the second time.
            process.Continue()
            self.assertTrue(process.GetState() == lldb.eStateStopped,
                            PROCESS_STOPPED)
            stack_traces2 = lldbutil.print_stacktraces(process, string_buffer=True)
            if self.TraceOn():
                print("Full stack traces when stopped on the breakpoint 'inner_inline' for the second time:")
                print(stack_traces2)
                self.expect(stack_traces2, "Second stop at %s:%d" % (self.source, self.second_stop), exe=False,
                            substrs = ['%s:%d' % (self.source, self.second_stop)])
