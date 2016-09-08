# TestObjCImportedTypes.py
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
"""n
Test that we are able to deal with ObjC-imported types
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftObjCImportedTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipUnlessDarwin
    @decorators.expectedFailureDarwin("rdar://15930675")
    def test_swift_objc_imported_types(self):
        """Test that we are able to deal with ObjC-imported types"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that we are able to deal with ObjC-imported types"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
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

        nss = self.frame.FindVariable("nss")
        nsn = self.frame.FindVariable("nsn")
        nsmo = self.frame.FindVariable("nsmo")
        nsmd = self.frame.FindVariable("nsmd")

        lldbutil.check_variable(
            self,
            nss,
            use_dynamic=False,
            typename="Foundation.NSString")
        lldbutil.check_variable(
            self,
            nsn,
            use_dynamic=False,
            typename="Foundation.NSNumber")
        lldbutil.check_variable(
            self,
            nsmo,
            use_dynamic=False,
            typename="CoreData.NSManagedObject")
        lldbutil.check_variable(
            self,
            nsmd,
            use_dynamic=False,
            typename="Foundation.NSMutableDictionary")

        # pending rdar://15798504, but not critical for the test
        #lldbutil.check_variable(self, nss, use_dynamic=True, summary='@"abc"')
        lldbutil.check_variable(self, nsn, use_dynamic=True, summary='(long)3')
        lldbutil.check_variable(
            self,
            nsmo,
            use_dynamic=True,
            typename='NSManagedObject *')
        lldbutil.check_variable(
            self,
            nsmd,
            use_dynamic=True,
            summary='1 key/value pair')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
