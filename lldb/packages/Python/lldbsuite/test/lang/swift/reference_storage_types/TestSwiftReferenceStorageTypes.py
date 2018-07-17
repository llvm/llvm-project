# TestSwiftReferenceStorageTypes.py
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
Test weak, unowned and unmanaged types
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftReferenceStorageTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_reference_storage_types(self):
        """Test weak, unowned and unmanaged types"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test weak, unowned and unmanaged types"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here',
            self.main_source_spec)
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

        self.expect('frame variable myclass.sub_001', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_002', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_003', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_004', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_005', substrs=['x = 1'])
        self.expect('frame variable myclass.sub_006', substrs=['x = 1'])

        self.expect('expression myclass.sub_001', substrs=['x = 1'])
        self.expect('expression myclass.sub_002', substrs=['x = 1'])
        self.expect('expression myclass.sub_003', substrs=['x = 1'])
        self.expect('expression myclass.sub_004', substrs=['x = 1'])
        self.expect('expression myclass.sub_005', substrs=['x = 1'])
        self.expect('expression myclass.sub_006', substrs=['x = 1'])

        self.expect('expression myclass.sub_001!', substrs=['x = 1'])
        self.expect('expression myclass.sub_002!', substrs=['x = 1'])
        self.expect('expression myclass.sub_003!', substrs=['x = 1'])
        self.expect('expression myclass.sub_004!', substrs=['x = 1'])

        myclass = self.frame.FindVariable("myclass")
        sub_001 = myclass.GetChildMemberWithName("sub_001")
        sub_002 = myclass.GetChildMemberWithName("sub_002")
        sub_003 = myclass.GetChildMemberWithName("sub_003")
        sub_004 = myclass.GetChildMemberWithName("sub_004")
        sub_005 = myclass.GetChildMemberWithName("sub_005")
        sub_006 = myclass.GetChildMemberWithName("sub_006")

        sub_001_type = sub_001.GetType()
        sub_002_type = sub_002.GetType()
        sub_003_type = sub_003.GetType()
        sub_004_type = sub_004.GetType()
        sub_005_type = sub_005.GetType()
        sub_006_type = sub_006.GetType()

        self.assertTrue(sub_001_type.IsValid(), "001.GetType() is valid")
        self.assertTrue(sub_002_type.IsValid(), "002.GetType() is valid")
        self.assertTrue(sub_003_type.IsValid(), "003.GetType() is valid")
        self.assertTrue(sub_004_type.IsValid(), "004.GetType() is valid")
        self.assertTrue(sub_005_type.IsValid(), "005.GetType() is valid")
        self.assertTrue(sub_006_type.IsValid(), "006.GetType() is valid")

        sub_001_type.GetTypeClass()
        sub_002_type.GetTypeClass()
        sub_003_type.GetTypeClass()
        sub_004_type.GetTypeClass()
        sub_005_type.GetTypeClass()
        sub_006_type.GetTypeClass()

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
