"""Test the go expression parser/interpreter."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGoUserExpression(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @skipIfRemote  # Not remote test suit ready
    @skipUnlessGoInstalled
    def test_with_dsym_and_python_api(self):
        """Test GoASTUserExpress."""
        self.buildGo()
        self.launchProcess()
        self.go_expressions()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.go"
        self.break_line = line_number(
            self.main_source, '// Set breakpoint here.')

    def check_builtin(self, name, size=0, typeclass=lldb.eTypeClassBuiltin):
        tl = self.target().FindTypes(name)
        self.assertEqual(1, len(tl))
        t = list(tl)[0]
        self.assertEqual(name, t.name)
        self.assertEqual(typeclass, t.type)
        if size > 0:
            self.assertEqual(size, t.size)

    def launchProcess(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bpt = target.BreakpointCreateByLocation(
            self.main_source, self.break_line)
        self.assertTrue(bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(process, bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertTrue(len(thread_list) == 1,
                        "More than one thread stopped at our breakpoint.")

        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

    def go_expressions(self):
        frame = self.frame()
        v = frame.EvaluateExpression("1")
        self.assertEqual(1, v.GetValueAsSigned())
        x = frame.EvaluateExpression("x")
        self.assertEqual(22, x.GetValueAsSigned())

        a = frame.EvaluateExpression("a")
        self.assertEqual(3, a.GetNumChildren())
        a0 = a.GetChildAtIndex(0)
        self.assertEqual(8, a0.GetValueAsSigned())

        # Array indexing
        a0 = frame.EvaluateExpression("a[0]")
        self.assertEqual(8, a0.GetValueAsSigned())

        # Slice indexing
        b1 = frame.EvaluateExpression("b[1]")
        self.assertEqual(9, b1.GetValueAsSigned())

        # Test global in this package
        g = frame.EvaluateExpression("myGlobal")
        self.assertEqual(17, g.GetValueAsSigned(), str(g))

        # Global with package name
        g = frame.EvaluateExpression("main.myGlobal")
        self.assertEqual(17, g.GetValueAsSigned(), str(g))

        # Global with quoted package name
        g = frame.EvaluateExpression('"main".myGlobal')
        self.assertEqual(17, g.GetValueAsSigned(), str(g))

        # Casting with package local type
        s = frame.EvaluateExpression("*(*myStruct)(i.data)")
        sb = s.GetChildMemberWithName("a")
        self.assertEqual(2, sb.GetValueAsSigned())

        # casting with explicit package
        s = frame.EvaluateExpression("*(*main.myStruct)(i.data)")
        sb = s.GetChildMemberWithName("a")
        self.assertEqual(2, sb.GetValueAsSigned())

        # Casting quoted package
        s = frame.EvaluateExpression('*(*"main".myStruct)(i.data)')
        sb = s.GetChildMemberWithName("b")
        self.assertEqual(-1, sb.GetValueAsSigned())

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
