"""Test calling functions in static methods with a stripped binary."""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestObjCStaticMethodStripped(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "static.m"
        self.break_line = line_number(self.main_source, '// Set breakpoint here.')

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    @skipIfDwarf    # This test requires a stripped binary and a dSYM
    @skipIfDWO      # This test requires a stripped binary and a dSYM
    #<rdar://problem/12042992>
    def test_with_python_api(self):
        """Test calling functions in static methods with a stripped binary."""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()
        exe = os.path.join(os.getcwd(), "a.out.stripped")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bpt = target.BreakpointCreateByLocation(self.main_source, self.break_line)
        self.assertTrue(bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue (len(thread_list) != 0, "No thread stopped at our breakpoint.")
        self.assertTrue (len(thread_list) == 1, "More than one thread stopped at our breakpoint.")
            
        # Now make sure we can call a function in the static method we've stopped in.
        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue (frame, "Got a valid frame 0 frame.")

        cmd_value = frame.EvaluateExpression ("(char *) sel_getName (_cmd)")
        self.assertTrue (cmd_value.IsValid())
        sel_name = cmd_value.GetSummary()
        self.assertTrue (sel_name == "\"doSomethingWithString:\"", "Got the right value for the selector as string.")

        cmd_value = frame.EvaluateExpression ("[Foo doSomethingElseWithString:string]")
        self.assertTrue (cmd_value.IsValid())
        string_length = cmd_value.GetValueAsUnsigned()
        self.assertTrue (string_length == 27, "Got the right value from another class method on the same class.")
