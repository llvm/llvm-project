# TestObjCIvarDiscovery.py
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
Test that we can correctly see ivars from the Objective-C runtime
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import commands
import re
import unittest2
import shutil

class TestObjCIVarDiscovery(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(bugnumber="rdar://45929100")
    def test_nodbg(self):
        self.build()
        shutil.rmtree(self.getBuildArtifact("aTestFramework.framework.dSYM"))
        self.do_test(False)

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(bugnumber="rdar://45929100")
    def test_dbg(self):
        self.build()
        self.do_test(True)

    def prepare_value(self, value):
        value.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        value.SetPreferSyntheticValue(True)
        return value

    def do_test(self, dbg):
        """Test that we can correctly see ivars from the Objective-C runtime"""
        exe_name = "a.out"
        src_main = lldb.SBFileSpec("main.swift")
        exe = self.getBuildArtifact(exe_name)

        if dbg:
            self.runCmd("type category disable runtime-synthetics")

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', src_main)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        envp = ['DYLD_FRAMEWORK_PATH=.']
        process = target.LaunchSimple(None, envp, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        if dbg:
            self.expect(
                "image list", "Contents/Resources/DWARF/aTestFramework")
        else:
            self.expect(
                "image list",
                "Contents/Resources/DWARF/aTestFramework",
                matching=False)

        self.runCmd("frame variable -d run --show-types --ptr-depth=1")

        obj = self.prepare_value(self.frame.FindVariable("object"))

        mysubclass = self.prepare_value(obj.GetChildAtIndex(0))
        myclass = self.prepare_value(mysubclass.GetChildAtIndex(0))

        m_pair = myclass.GetChildMemberWithName("m_pair")
        m_pair_A = m_pair.GetChildMemberWithName("A")
        m_pair_B = m_pair.GetChildMemberWithName("B")

        self.assertEqual(m_pair_A.GetValueAsUnsigned(), 1)
        self.assertEqual(m_pair_B.GetValueAsUnsigned(), 2)

        m_derived = self.prepare_value(
            myclass.GetChildMemberWithName("m_base"))

        m_derivedX = m_derived.GetChildMemberWithName("m_DerivedX")

        self.assertEqual(m_derivedX.GetValueAsUnsigned(), 1)

        m_numbers = self.prepare_value(
            myclass.GetChildMemberWithName("m_myclass_numbers"))

        self.assertTrue(
            m_numbers.GetSummary() == '"3 values"',
            "m_myclass_numbers != 3 values")

        m_subclass_ivar = mysubclass.GetChildMemberWithName("m_subclass_ivar")
        self.assertTrue(
            m_subclass_ivar.GetValueAsUnsigned() == 42,
            "m_subclass_ivar != 42")

        m_mysubclass_s = mysubclass.GetChildMemberWithName("m_mysubclass_s")
        self.assertTrue(
            m_mysubclass_s.GetSummary() == '"an NSString here"',
            'm_subclass_s != "an NSString here"')

        m_mysubclass_r = mysubclass.GetChildMemberWithName("m_mysubclass_r")
        self.assertTrue(
            re.search(
                '.*x=0[, ]+y=0.*width=30[, ]+height=40.*',
                m_mysubclass_r.GetSummary()
            ) is not None,
            'm_subclass_r != origin=(x=0, y=0) size=(width=30, height=40)')

        swiftivar = obj.GetChildMemberWithName("swiftivar")
        self.assertTrue(
            swiftivar.GetSummary() == '"Hey Swift!"', "swiftivar != Hey Swift")

        silly = self.prepare_value(obj.GetChildMemberWithName("silly"))

        silly_x = silly.GetChildMemberWithName("x")
        silly_url = silly.GetChildMemberWithName("url")

        self.assertTrue(silly_x.GetValueAsUnsigned() == 12, "x != 12")
        self.assertTrue(
            silly_url.GetSummary() == '"http://www.apple.com"',
            "url != apple.com")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
