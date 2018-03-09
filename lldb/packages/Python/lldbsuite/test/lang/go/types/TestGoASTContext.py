"""Test the go DWARF type parsing."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGoASTContext(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @skipIfFreeBSD  # llvm.org/pr24895 triggers assertion failure
    @skipIfRemote  # Not remote test suit ready
    @no_debug_info_test
    @skipUnlessGoInstalled
    @expectedFailureAll(bugnumber="llvm.org/pr33643")
    def test_with_dsym_and_python_api(self):
        """Test GoASTContext dwarf parsing."""
        self.buildGo()
        self.launchProcess()
        self.go_builtin_types()
        self.check_main_vars()

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
        exe = self.getBuildArtifact("a.out")

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

    def go_builtin_types(self):
        address_size = self.target().GetAddressByteSize()
        self.check_builtin('bool')
        self.check_builtin('uint8', 1)
        self.check_builtin('int8', 1)
        self.check_builtin('uint16', 2)
        self.check_builtin('int16', 2)
        self.check_builtin('uint32', 4)
        self.check_builtin('int32', 4)
        self.check_builtin('uint64', 8)
        self.check_builtin('int64', 8)
        self.check_builtin('uintptr', address_size)
        self.check_builtin('int', address_size)
        self.check_builtin('uint', address_size)
        self.check_builtin('float32', 4)
        self.check_builtin('float64', 8)
        self.check_builtin('complex64', 8, lldb.eTypeClassComplexFloat)
        self.check_builtin('complex128', 16, lldb.eTypeClassComplexFloat)

    def var(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid(), "%s %s" % (VALID_VARIABLE, name))
        return var

    def check_main_vars(self):
        v = self.var('theBool')
        self.assertEqual('true', v.value)

        v = self.var('theInt')
        self.assertEqual('-7', v.value)

        v = self.var('theComplex')
        self.assertEqual('1 + 2i', v.value)

        v = self.var('thePointer')
        self.assertTrue(v.TypeIsPointerType())
        self.assertEqual('-10', v.Dereference().value)
        self.assertEqual(1, v.GetNumChildren())
        self.assertEqual('-10', v.GetChildAtIndex(0).value)

        # print()
        # print(os.getpid())
        # time.sleep(60)
        v = self.var('theStruct')
        if v.TypeIsPointerType():
            v = v.Dereference()
        self.assertEqual(2, v.GetNumChildren())
        self.assertEqual('7', v.GetChildAtIndex(0).value)
        self.assertEqual('7', v.GetChildMemberWithName('myInt').value)
        self.assertEqual(
            v.load_addr,
            v.GetChildAtIndex(1).GetValueAsUnsigned())
        self.assertEqual(v.load_addr, v.GetChildMemberWithName(
            'myPointer').GetValueAsUnsigned())

        # Test accessing struct fields through pointers.
        v = v.GetChildMemberWithName('myPointer')
        self.assertTrue(v.TypeIsPointerType())
        self.assertEqual(2, v.GetNumChildren())
        self.assertEqual('7', v.GetChildAtIndex(0).value)
        c = v.GetChildMemberWithName('myPointer')
        self.assertTrue(c.TypeIsPointerType())
        self.assertEqual(2, c.GetNumChildren())
        self.assertEqual('7', c.GetChildAtIndex(0).value)

        v = self.var('theArray')
        self.assertEqual(5, v.GetNumChildren())
        for i in list(range(5)):
            self.assertEqual(str(i + 1), v.GetChildAtIndex(i).value)
