# TestSubmoduleImport.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Tests that the expression parser can auto-import and hand-import sub-modules
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftSubmoduleImport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Have to find some submodule that is present on both Darwin & Linux for this
    # test to run on both systems...

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test_swift_submodule_import(self):
        """Tests that swift expressions can import sub-modules correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set a breakpoint here', lldb.SBFileSpec('main.swift'))

        options = lldb.SBExpressionOptions()
        options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)

        # We'll be asked to auto-import Darwin.C when we evaluate this expression,
        # so even though it doesn't seem like it this does test auto-import:
        value = self.frame().EvaluateExpression("b", options)
        self.assertTrue(value.IsValid(), "Got a valid variable back from b")
        self.assertTrue(value.GetError().Success(),
                        "And the variable was successfully evaluated")
        result = value.GetSummary()
        self.assertTrue(
            result == '"aa"',
            "And the variable's value was correct.")

        # Now make sure we can explicitly do the import:
        value = self.frame().EvaluateExpression('import Darwin.C\n b', options)
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
