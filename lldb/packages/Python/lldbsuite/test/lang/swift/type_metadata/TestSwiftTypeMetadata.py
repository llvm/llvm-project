# TestSwiftTypeMetadata.py
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
Test that LLDB can effectively use the type metadata to reconstruct dynamic types for Swift
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftTypeMetadataTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_type_metadata(self):
        """Test that LLDB can effectively use the type metadata to reconstruct dynamic types for Swift"""
        self.build()
        self.var_commands()

    def setUp(self):
        TestBase.setUp(self)

    def var_commands(self):
        """Test that LLDB can effectively use the type metadata to reconstruct dynamic types for Swift"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_source_regexp(self, "// Set breakpoint here")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("frame select 0", substrs=['foo', 'x', 'ivar'])
        self.expect(
            "frame variable -d run -- x",
            substrs=[
                '(a.AClass) x',
                'ivar = 3735928559'])  # first stop on foo
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['bar', 'x', 'y'])
        self.expect(
            "frame variable -d run -- x y",
            substrs=[
                '(Int64) x',
                '(Float) y'])  # first stop on bar
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['foo', 'x'])
        self.expect(
            "frame variable -d run -- x",
            substrs=[
                '(a.AClass) x',
                'ivar = 3735928559'])  # second stop on foo
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['baz', 'x'])
        self.expect(
            "frame variable -d run -- x",
            substrs=[
                '(a.AClass) x',
                'ivar = 3735928559'])  # first stop on baz
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['bar', 'x'])
        self.expect(
            "frame variable -d run -- x y",
            substrs=[
                '(a.AClass) x',
                '(a.AClass) y'])  # second stop on bar
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['bat', 'x'])
        self.expect("frame variable -d run -- x",
                    substrs=['(a.ADerivedClass) x'])  # first stop on bat
        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame select 0", substrs=['bat', 'x'])
        self.expect(
            "frame variable -d run -- x",
            substrs=['(a.AnotherDerivedClass) x'])  # second stop on bat
        self.runCmd("continue", RUN_SUCCEEDED)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
