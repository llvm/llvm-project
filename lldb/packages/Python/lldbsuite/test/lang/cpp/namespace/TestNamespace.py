"""
Test the printing of anonymous and named namespace variables.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NamespaceBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(bugnumber="llvm.org/pr28548", compiler="gcc")
    @expectedFailureAll(oslist=["windows"])
    def test_breakpoints_func_auto(self):
        """Test that we can set breakpoints correctly by basename to find all functions whose basename is "func"."""
        self.build()

        names = [
            "func()",
            "func(int)",
            "A::B::func()",
            "A::func()",
            "A::func(int)"]

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        module_list = lldb.SBFileSpecList()
        module_list.Append(lldb.SBFileSpec(exe, False))
        cu_list = lldb.SBFileSpecList()
        # Set a breakpoint by name "func" which should pick up all functions
        # whose basename is "func"
        bp = target.BreakpointCreateByName(
            "func", lldb.eFunctionNameTypeAuto, module_list, cu_list)
        for bp_loc in bp:
            name = bp_loc.GetAddress().GetFunction().GetName()
            self.assertTrue(
                name in names,
                "make sure breakpoint locations are correct for 'func' with eFunctionNameTypeAuto")

    @expectedFailureAll(bugnumber="llvm.org/pr28548", compiler="gcc")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_breakpoints_func_full(self):
        """Test that we can set breakpoints correctly by fullname to find all functions whose fully qualified name is "func"
           (no namespaces)."""
        self.build()

        names = ["func()", "func(int)"]

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        module_list = lldb.SBFileSpecList()
        module_list.Append(lldb.SBFileSpec(exe, False))
        cu_list = lldb.SBFileSpecList()

        # Set a breakpoint by name "func" whose fullly qualified named matches "func" which
        # should pick up only functions whose basename is "func" and has no
        # containing context
        bp = target.BreakpointCreateByName(
            "func", lldb.eFunctionNameTypeFull, module_list, cu_list)
        for bp_loc in bp:
            name = bp_loc.GetAddress().GetFunction().GetName()
            self.assertTrue(
                name in names,
                "make sure breakpoint locations are correct for 'func' with eFunctionNameTypeFull")

    def test_breakpoints_a_func_full(self):
        """Test that we can set breakpoints correctly by fullname to find all functions whose fully qualified name is "A::func"."""
        self.build()

        names = ["A::func()", "A::func(int)"]

        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        module_list = lldb.SBFileSpecList()
        module_list.Append(lldb.SBFileSpec(exe, False))
        cu_list = lldb.SBFileSpecList()

        # Set a breakpoint by name "A::func" whose fullly qualified named matches "A::func" which
        # should pick up only functions whose basename is "func" and is
        # contained in the "A" namespace
        bp = target.BreakpointCreateByName(
            "A::func", lldb.eFunctionNameTypeFull, module_list, cu_list)
        for bp_loc in bp:
            name = bp_loc.GetAddress().GetFunction().GetName()
            self.assertTrue(
                name in names,
                "make sure breakpoint locations are correct for 'A::func' with eFunctionNameTypeFull")


class NamespaceTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for declarations of namespace variables i and
        # j.
        self.line_var_i = line_number(
            'main.cpp', '// Find the line number for anonymous namespace variable i.')
        self.line_var_j = line_number(
            'main.cpp', '// Find the line number for named namespace variable j.')
        # And the line number to break at.
        self.line_break = line_number('main.cpp',
                                      '// Set break point at this line.')
        # Break inside do {} while and evaluate value
        self.line_break_ns1 = line_number('main.cpp', '// Evaluate ns1::value')
        self.line_break_ns2 = line_number('main.cpp', '// Evaluate ns2::value')

    def runToBkpt(self, command):
        self.runCmd(command, RUN_SUCCEEDED)
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

    # rdar://problem/8668674
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    def test_with_run_command(self):
        """Test that anonymous and named namespace variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.line_break_ns1,
            num_expected_locations=1,
            loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.line_break_ns2,
            num_expected_locations=1,
            loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.line_break,
            num_expected_locations=1,
            loc_exact=True)

        self.runToBkpt("run")
        # Evaluate ns1::value
        self.expect("expression -- value", startstr="(int) $0 = 100")

        self.runToBkpt("continue")
        # Evaluate ns2::value
        self.expect("expression -- value", startstr="(int) $1 = 200")

        self.runToBkpt("continue")
        # On Mac OS X, gcc 4.2 emits the wrong debug info with respect to
        # types.
        slist = ['(int) a = 12', 'anon_uint', 'a_uint', 'b_uint', 'y_uint']
        if self.platformIsDarwin() and self.getCompiler() in [
                'clang', 'llvm-gcc']:
            slist = ['(int) a = 12',
                     '::my_uint_t', 'anon_uint = 0',
                     '(A::uint_t) a_uint = 1',
                     '(A::B::uint_t) b_uint = 2',
                     '(Y::uint_t) y_uint = 3']

        # 'frame variable' displays the local variables with type information.
        self.expect('frame variable', VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=slist)

        # 'frame variable' with basename 'i' should work.
        self.expect(
            "frame variable --show-declaration --show-globals i",
            startstr="main.cpp:%d: (int) (anonymous namespace)::i = 3" %
            self.line_var_i)
        # main.cpp:12: (int) (anonymous namespace)::i = 3

        # 'frame variable' with basename 'j' should work, too.
        self.expect(
            "frame variable --show-declaration --show-globals j",
            startstr="main.cpp:%d: (int) A::B::j = 4" %
            self.line_var_j)
        # main.cpp:19: (int) A::B::j = 4

        # 'frame variable' should support address-of operator.
        self.runCmd("frame variable &i")

        # 'frame variable' with fully qualified name 'A::B::j' should work.
        self.expect("frame variable A::B::j", VARIABLES_DISPLAYED_CORRECTLY,
                    startstr='(int) A::B::j = 4',
                    patterns=[' = 4'])

        # So should the anonymous namespace case.
        self.expect(
            "frame variable '(anonymous namespace)::i'",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int) (anonymous namespace)::i = 3',
            patterns=[' = 3'])

        # rdar://problem/8660275
        # test/namespace: 'expression -- i+j' not working
        # This has been fixed.
        self.expect("expression -- i + j",
                    startstr="(int) $2 = 7")
        # (int) $2 = 7

        self.runCmd("expression -- i")
        self.runCmd("expression -- j")

        # rdar://problem/8668674
        # expression command with fully qualified namespace for a variable does
        # not work
        self.expect("expression -- ::i", VARIABLES_DISPLAYED_CORRECTLY,
                    patterns=[' = 3'])
        self.expect("expression -- A::B::j", VARIABLES_DISPLAYED_CORRECTLY,
                    patterns=[' = 4'])

        # expression command with function in anonymous namespace
        self.expect("expression -- myanonfunc(3)",
                    patterns=[' = 6'])

        # global namespace qualification with function in anonymous namespace
        self.expect("expression -- ::myanonfunc(4)",
                    patterns=[' = 8'])

        self.expect("p myanonfunc",
                    patterns=['\(anonymous namespace\)::myanonfunc\(int\)'])

        self.expect("p variadic_sum", patterns=[
                    '\(anonymous namespace\)::variadic_sum\(int, ...\)'])
