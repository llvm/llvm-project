# TestSwiftHashedContainerEnum.py

"""
Test combinations of hashed swift containers with enums as keys/values
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftHashedContainerEnum(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @swiftTest
    @add_test_categories(["swiftpr"])
    def test_any_object_type(self):
        """Test combinations of hashed swift containers with enums"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test combinations of hashed swift containers with enums"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            '// break here', self.main_source_spec)
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

        self.expect(
            'frame variable -d run -- testA',
            substrs=[
                'key = c',
                'value = 1',
                'key = b',
                'value = 2'])
        self.expect(
            'expr -d run -- testA',
            substrs=[
                'key = c',
                'value = 1',
                'key = b',
                'value = 2'])

        self.expect(
            'frame variable -d run -- testB',
            substrs=[
                'key = "a", value = 1',
                'key = "b", value = 2'])
        self.expect(
            'expr -d run -- testB',
            substrs=[
                'key = "a", value = 1',
                'key = "b", value = 2'])

        self.expect(
            'frame variable -d run -- testC',
            substrs=['key = b', 'value = 2'])
        self.expect(
            'expr -d run -- testC',
            substrs=['key = b', 'value = 2'])

        self.expect(
            'frame variable -d run -- testD',
            substrs=['[0] = c'])
        self.expect(
            'expr -d run -- testD',
            substrs=['[0] = c'])
