# encoding: utf-8
"""
Test lldb Obj-C exception support.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCExceptionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_objc_exceptions_1(self):
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_to_name_breakpoint(self, "objc_exception_throw")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        self.expect('thread exception', substrs=[
                '(NSException *) exception = ',
                'name: "ThrownException" - reason: "SomeReason"',
            ])

        lldbutil.run_to_source_breakpoint(self, "// Set break point at this line.", lldb.SBFileSpec("main.m"))

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        target = self.dbg.GetSelectedTarget()
        thread = target.GetProcess().GetSelectedThread()
        frame = thread.GetSelectedFrame()

        self.expect(
            'frame variable e1',
            substrs=[
                '(NSException *) e1 = ',
                'name: @"ExceptionName" - reason: @"SomeReason"'
            ])

        self.expect(
            'frame variable --dynamic-type no-run-target *e1',
            substrs=[
                '(NSException) *e1 = ',
                'name = ', '@"ExceptionName"',
                'reason = ', '@"SomeReason"',
                'userInfo = ', '1 key/value pair',
                'reserved = ', 'nil',
            ])

        e1 = frame.FindVariable("e1")
        self.assertTrue(e1)
        self.assertEqual(e1.type.name, "NSException *")
        self.assertEqual(e1.GetSummary(), 'name: @"ExceptionName" - reason: @"SomeReason"')
        self.assertEqual(e1.GetChildMemberWithName("name").description, "ExceptionName")
        self.assertEqual(e1.GetChildMemberWithName("reason").description, "SomeReason")
        userInfo = e1.GetChildMemberWithName("userInfo").dynamic
        self.assertEqual(userInfo.summary, "1 key/value pair")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(0).description, "some_key")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(1).description, "some_value")
        self.assertEqual(e1.GetChildMemberWithName("reserved").description, "<nil>")

        self.expect(
            'frame variable e2',
            substrs=[
                '(NSException *) e2 = ',
                'name: @"ThrownException" - reason: @"SomeReason"'
            ])

        self.expect(
            'frame variable --dynamic-type no-run-target *e2',
            substrs=[
                '(NSException) *e2 = ',
                'name = ', '@"ThrownException"',
                'reason = ', '@"SomeReason"',
                'userInfo = ', '1 key/value pair',
                'reserved = ',
            ])

        e2 = frame.FindVariable("e2")
        self.assertTrue(e2)
        self.assertEqual(e2.type.name, "NSException *")
        self.assertEqual(e2.GetSummary(), 'name: @"ThrownException" - reason: @"SomeReason"')
        self.assertEqual(e2.GetChildMemberWithName("name").description, "ThrownException")
        self.assertEqual(e2.GetChildMemberWithName("reason").description, "SomeReason")
        userInfo = e2.GetChildMemberWithName("userInfo").dynamic
        self.assertEqual(userInfo.summary, "1 key/value pair")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(0).description, "some_key")
        self.assertEqual(userInfo.GetChildAtIndex(0).GetChildAtIndex(1).description, "some_value")
        reserved = e2.GetChildMemberWithName("reserved").dynamic
        self.assertGreater(reserved.num_children, 0)
        callStackReturnAddresses = [reserved.GetChildAtIndex(i).GetChildAtIndex(1) for i in range(0, reserved.GetNumChildren())
                if reserved.GetChildAtIndex(i).GetChildAtIndex(0).description == "callStackReturnAddresses"][0].dynamic
        children = [callStackReturnAddresses.GetChildAtIndex(i) for i in range(0, callStackReturnAddresses.num_children)]

        pcs = [i.unsigned for i in children]
        names = [target.ResolveSymbolContextForAddress(lldb.SBAddress(pc, target), lldb.eSymbolContextSymbol).GetSymbol().name for pc in pcs]
        for n in ["objc_exception_throw", "foo", "main"]:
            self.assertTrue(n in names, "%s is in the exception backtrace (%s)" % (n, names))
