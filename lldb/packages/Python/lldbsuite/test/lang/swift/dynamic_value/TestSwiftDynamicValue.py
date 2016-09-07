# TestSwiftDynamicValue.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Tests that dynamic values work correctly for Swift
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftDynamicValueTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_dynamic_value(self):
        """Tests that dynamic values work correctly for Swift"""
        self.build()
        self.dynamic_val_commands()

    def setUp(self):
        TestBase.setUp(self)

    def dynamic_val_commands(self):
        """Tests that dynamic values work correctly for Swift"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_source_regexp(
            self, "// Set a breakpoint here")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect(
            "frame variable",
            substrs=[
                "AWrapperClass) aWrapper",
                "SomeClass) anItem = ",
                "x = ",
                "Base<Int>) aBase = 0x",
                "v = 449493530"])
        self.expect(
            "frame variable -d run --show-types",
            substrs=[
                "AWrapperClass) aWrapper",
                "YetAnotherClass) anItem = ",
                "x = ",
                "y = ",
                "z = ",
                "Derived<Int>) aBase = 0x",
                "Base<Int>)",
                ".Base = {",
                "v = 449493530",
                "q = 3735928559"])
        self.runCmd("continue")
        self.expect(
            "frame variable",
            substrs=[
                "AWrapperClass) aWrapper",
                "SomeClass) anItem = ",
                "x = ",
                "Base<Int>) aBase = 0x",
                "v = 449493530"])
        self.expect(
            "frame variable -d run --show-types",
            substrs=[
                "AWrapperClass) aWrapper",
                "YetAnotherClass) anItem = ",
                "x = ",
                "y = ",
                "z = ",
                "Derived<Int>) aBase = 0x",
                "Base<Int>)",
                ".Base = {",
                "v = 449493530",
                "q = 3735928559"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
