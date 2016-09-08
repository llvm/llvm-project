# TestSubmoduleImport.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Tests that the expression parser can auto-import and hand-import sub-modules
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftSubmoduleImport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    def test_swift_submodule_import(self):
        """Tests that swift expressions can import sub-modules correctly"""
        self.build()
        self.do_test()

    # Have to find some submodule that is present on both Darwin & Linux for this
    # test to run on both systems...

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Tests that swift expressions resolve scoped variables correctly"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        global_scope_bkpt = target.BreakpointCreateBySourceRegex(
            'Set a breakpoint here', self.main_source_spec)
        self.assertTrue(
            global_scope_bkpt.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, global_scope_bkpt)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        options = lldb.SBExpressionOptions()
        options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)

        # We'll be asked to auto-import Darwin.C when we evaluate this expression,
        # so even though it doesn't seem like it this does test auto-import:
        value = self.frame.EvaluateExpression("b", options)
        self.assertTrue(value.IsValid(), "Got a valid variable back from b")
        self.assertTrue(value.GetError().Success(),
                        "And the variable was successfully evaluated")
        result = value.GetSummary()
        self.assertTrue(
            result == '"aa"',
            "And the variable's value was correct.")

        # Now make sure we can explicitly do the import:
        value = self.frame.EvaluateExpression('import Darwin.C\n b', options)
        self.assertTrue(
            value.IsValid(),
            "Got a valid value back from import Darwin.C")
        self.assertTrue(
            value.GetError().Success(),
            "The import was not successful: %s" %
            (value.GetError().GetCString()))

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
