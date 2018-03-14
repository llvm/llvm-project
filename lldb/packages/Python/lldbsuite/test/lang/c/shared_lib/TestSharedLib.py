"""Test that types defined in shared libraries work correctly."""

from __future__ import print_function


import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class SharedLibTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def common_test_expr(self, preload_symbols):
        if "clang" in self.getCompiler() and "3.4" in self.getCompilerVersion():
            self.skipTest(
                "llvm.org/pr16214 -- clang emits partial DWARF for structures referenced via typedef")

        self.build()
        self.common_setup(preload_symbols)

        # This should display correctly.
        self.expect(
            "expression --show-types -- *my_foo_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(foo)",
                "(sub_foo)",
                "other_element = 3"])

        self.expect(
            "expression GetMeASubFoo(my_foo_ptr)",
            startstr="(sub_foo *) $")

    @expectedFailureAll(oslist=["windows"])
    def test_expr(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.common_test_expr(True)

    @expectedFailureAll(oslist=["windows"])
    def test_expr_no_preload(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable, but with preloading disabled"""
        self.common_test_expr(False)

    @unittest2.expectedFailure("llvm.org/PR36712")
    def test_frame_variable(self):
        """Test that types work when defined in a shared library and forward-declared in the main executable"""
        self.build()
        self.common_setup()

        # This should display correctly.
        self.expect(
            "frame variable --show-types -- *my_foo_ptr",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(foo)",
                "(sub_foo)",
                "other_element = 3"])

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = 'main.c'
        self.line = line_number(self.source, '// Set breakpoint 0 here.')
        self.shlib_names = ["foo"]

    def common_setup(self, preload_symbols = True):
        # Run in synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.runCmd("settings set target.preload-symbols " + str(preload_symbols).lower())

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, self.source, self.line, num_expected_locations=1, loc_exact=True)

        # Register our shared libraries for remote targets so they get
        # automatically uploaded
        environment = self.registerSharedLibrariesWithTarget(
            target, self.shlib_names)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])
