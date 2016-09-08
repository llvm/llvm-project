# TestSwiftGetValueAsUnsigned.py
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
Tests that the SBValue::GetValueAsUnsigned() API works for Swift types
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftGetValueAsUnsignedAPITest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://problem/23426695")
    def test_get_value_as_unsigned_sbapi(self):
        """Tests that the SBValue::GetValueAsUnsigned() API works for Swift types"""
        self.build()
        self.getvalue_commands()

    def setUp(self):
        TestBase.setUp(self)

    def getvalue_commands(self):
        """Tests that the SBValue::GetValueAsUnsigned() API works for Swift types"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_source_regexp(
            self, r"break here", num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        frame = self.frame()
        string = frame.FindVariable("aString")
        number = frame.FindVariable("aNumber")
        number.SetPreferSyntheticValue(True)
        classobject = frame.FindVariable("aClassObject")

        numberValue = number.GetValueAsUnsigned()
        self.assertTrue(
            numberValue == 123456,
            "Swift.Int does not have a valid value")

        builtinPointerValue = string.GetChildMemberWithName(
            "str_value").GetChildMemberWithName("base").GetChildMemberWithName("value")
        self.assertTrue(builtinPointerValue != 0,
                        "Builtin.RawPointer does not have a valid value")

        objectPointerValue = string.GetChildMemberWithName(
            "str_value").GetChildMemberWithName("value")
        self.assertTrue(objectPointerValue != 0,
                        "Builtin.RawPointer does not have a valid value")

        classValue = classobject.GetValueAsUnsigned()
        self.assertTrue(
            classValue != 0,
            "Class types are aggregates with pointer values")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()
