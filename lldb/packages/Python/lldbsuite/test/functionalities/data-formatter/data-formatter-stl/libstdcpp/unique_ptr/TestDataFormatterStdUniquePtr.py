"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdUniquePtrDataFormatterTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD
    @skipIfWindows  # libstdcpp not ported to Windows
    @skipIfDarwin  # doesn't compile on Darwin
    def test_with_run_command(self):
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(
            self, "Set break point at this line.")
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        frame = self.frame()
        self.assertTrue(frame.IsValid())

        self.expect("frame variable nup", substrs=['nup = nullptr'])
        self.expect("frame variable iup", substrs=['iup = 0x', 'object = 123'])
        self.expect("frame variable sup", substrs=['sup = 0x', 'object = "foobar"'])

        self.expect("frame variable ndp", substrs=['ndp = nullptr'])
        self.expect("frame variable idp", substrs=['idp = 0x', 'object = 456', 'deleter = ', 'a = 1', 'b = 2'])
        self.expect("frame variable sdp", substrs=['sdp = 0x', 'object = "baz"', 'deleter = ', 'a = 3', 'b = 4'])
        
        self.assertEqual(123, frame.GetValueForVariablePath("iup.object").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("iup.deleter").IsValid())

        self.assertEqual('"foobar"', frame.GetValueForVariablePath("sup.object").GetSummary())
        self.assertFalse(frame.GetValueForVariablePath("sup.deleter").IsValid())

        self.assertEqual(456, frame.GetValueForVariablePath("idp.object").GetValueAsUnsigned())
        self.assertEqual('"baz"', frame.GetValueForVariablePath("sdp.object").GetSummary())

        idp_deleter = frame.GetValueForVariablePath("idp.deleter")
        self.assertTrue(idp_deleter.IsValid())
        self.assertEqual(1, idp_deleter.GetChildMemberWithName("a").GetValueAsUnsigned())
        self.assertEqual(2, idp_deleter.GetChildMemberWithName("b").GetValueAsUnsigned())

        sdp_deleter = frame.GetValueForVariablePath("sdp.deleter")
        self.assertTrue(sdp_deleter.IsValid())
        self.assertEqual(3, sdp_deleter.GetChildMemberWithName("a").GetValueAsUnsigned())
        self.assertEqual(4, sdp_deleter.GetChildMemberWithName("b").GetValueAsUnsigned())
