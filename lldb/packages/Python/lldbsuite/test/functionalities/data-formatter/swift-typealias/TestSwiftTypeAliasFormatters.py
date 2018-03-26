# TestSwiftTypeAliasFormatters.py
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
Test that Swift typealiases get formatted properly
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftTypeAliasFormatters(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_type_alias_formatters(self):
        """Test that Swift typealiases get formatted properly"""
        self.build()
        self.do_test()

    def do_test(self):
        """Test that Swift typealiases get formatted properly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        self.addTearDownHook(cleanup)

        self.expect("frame variable f", substrs=['Foo) f = (value = 12)'])
        self.expect("frame variable b", substrs=['Bar) b = (value = 24)'])

        self.runCmd('type summary add Foo -v -s "hello"')
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = hello'])

        self.runCmd('type summary add Bar -v -s "hi"')
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = hi'])

        self.runCmd("type summary delete Foo")
        self.expect("frame variable f", substrs=['Foo) f = (value = 12)'])
        self.expect("frame variable b", substrs=['Bar) b = hi'])

        self.runCmd("type summary delete Bar")
        self.runCmd("type summary add -C no -v Foo -s hello")
        self.expect("frame variable f", substrs=['Foo) f = hello'])
        self.expect("frame variable b", substrs=['Bar) b = (value = 24)'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
