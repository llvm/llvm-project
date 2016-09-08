# TestSwiftTypeLookup.py
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
Test the ability to look for type definitions at the command line
"""
import commands
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftTypeLookup(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_type_lookup(self):
        """Test the ability to look for type definitions at the command line"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test the ability to look for type definitions at the command line"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

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

        # for now, type lookup won't load the AST context, so force load it
        # before testing
        self.runCmd("expr 1")

        # check that basic cases work
        self.expect(
            "type lookup String",
            substrs=[
                'struct String {',
                'extension String : '])
        self.expect(
            "type lookup Cla1",
            substrs=[
                'class Cla1 {',
                'func bat(_ x: Swift.Int, y: a.Str1) -> Swift.Int'])
        self.expect(
            "type lookup Str1",
            substrs=[
                'struct Str1 {',
                'func bar()',
                'var b'])

        # check that specifiers are honored
        # self.expect('type lookup class Cla1', substrs=['class Cla1 {'])
        # self.expect('type lookup struct Cla1', substrs=['class Cla1 {'], matching=False, error=True)

        # check that modules are honored
        self.expect("type lookup Swift.String", substrs=['struct String {'])
        self.expect(
            "type lookup a.String",
            substrs=['struct String {'],
            matching=False)

        # check that a combination of module and specifier is honored
        # self.expect('type lookup class a.Cla1', substrs=['class Cla1 {'])
        # self.expect('type lookup class Swift.Cla1', substrs=['class Cla1 {'], matching=False, error=True)
        # self.expect('type lookup struct a.Cla1', substrs=['class Cla1 {'], matching=False, error=True)
        # self.expect('type lookup struct Swift.Cla1', substrs=['class Cla1 {'], matching=False, error=True)

        # check that nested types are looked up correctly
        self.expect('type lookup Toplevel.Nested.Deeper',
                    substrs=['class Deeper', 'func foo'])

        # check that mangled name lookup works
        self.expect(
            'type lookup _TtSi',
            substrs=[
                'struct Int',
                'extension Int'])

        # check that we can look for generic things
        self.expect('type lookup Generic', substrs=['class Generic', 'foo'])
        self.expect(
            'type lookup Generic<String>',
            substrs=[
                'class Generic',
                'func foo'])

        # check that we print comment text (and let you get rid of it)
        self.expect(
            'type lookup Int',
            substrs=['Create an instance'],
            matching=False)
        self.expect(
            'type lookup --show-help -- Int',
            substrs=['Create an instance'],
            matching=True)
        self.expect(
            'type lookup foo',
            substrs=[
                'func foo',
                'Int',
                'Double'],
            matching=True)
        self.expect(
            'type lookup --show-help -- print',
            substrs=[
                '/// ',
                'func print'],
            matching=True)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
