# TestSwiftArrayType.py
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
Check formatting for Swift.Array<T>
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftArrayType(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @decorators.swiftTest
    @decorators.skipIf(bugnumber='rdar://30663811', oslist=['linux'])
    @decorators.add_test_categories(["swiftpr"])
    def test_array(self):
        """Check formatting for Swift.Array<T>"""
        self.build()
        self.do_test()

    def do_test(self):
        """Check formatting for Swift.Array<T>"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0, lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        self.expect(
            "frame variable arrint",
            substrs=["([Int]) arrint = 0 values {}"])
        self.runCmd("continue")

        self.expect(
            "frame variable arrint",
            substrs=[
                "([Int]) arrint = 5 values {", "[0] = 1",
                "[0] = 1", "[1] = 2", "[2] = 3", "[3] = 4", "[4] = 5"])
        self.expect(
            "frame variable arrfoo",
            substrs=["Foo]) arrfoo = 0 values {}"])
        self.runCmd("continue")
        self.expect(
            "frame variable arrfoo",
            substrs=[
                "Foo]) arrfoo = 6 values {",
                "[0] = 0x", "x = 1",
                "[1] = 0x", "x = 2",
                "[2] = 0x", "x = 3",
                "[3] = 0x", "x = 4",
                "[4] = 0x", "x = 5",
                "[5] = 0x", "x = 6"])
        self.expect(
            "frame variable arrlar",
            substrs=["LargeDude]) arrlar = 0 values {}"])
        self.runCmd("continue")
        self.expect(
            "frame variable arrlar",
            substrs=[
                "LargeDude]) arrlar = 7 values {",
                '[0] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[1] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[2] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[3] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[4] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[5] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960',

                '[6] = ', 'x = 4294967295', 'y = 3735928559', 'z = 3203399405',
                'q = 267390960'])
        self.expect(
            "frame variable slice",
            substrs=[
                "Slice<Int>", "3 values {", "",
                "[1] = 2",
                "[2] = 3",
                "[3] = 4"])
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
