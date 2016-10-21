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


class StdSmartPtrDataFormatterTestCase(TestBase):
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

        self.expect("frame variable nsp", substrs=['nsp = nullptr'])
        self.expect("frame variable isp", substrs=['isp = 123', 'strong=1', 'weak=1'])
        self.expect("frame variable ssp", substrs=['ssp = "foobar"', 'strong=1', 'weak=1'])
        self.expect("frame variable nwp", substrs=['nwp = nullptr'])
        self.expect("frame variable iwp", substrs=['iwp = 123', 'strong=1', 'weak=1'])
        self.expect("frame variable swp", substrs=['swp = "foobar"', 'strong=1', 'weak=1'])
        
        frame = self.frame()
        self.assertTrue(frame.IsValid())

        self.assertEqual(0, frame.GetValueForVariablePath("nsp.pointer").GetValueAsUnsigned())
        self.assertEqual(0, frame.GetValueForVariablePath("nwp.pointer").GetValueAsUnsigned())

        self.assertNotEqual(0, frame.GetValueForVariablePath("isp.pointer").GetValueAsUnsigned())
        self.assertEqual(123, frame.GetValueForVariablePath("isp.object").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("isp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("isp.weak_count").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("isp.foobar").IsValid())

        self.assertNotEqual(0, frame.GetValueForVariablePath("ssp.pointer").GetValueAsUnsigned())
        self.assertEqual('"foobar"', frame.GetValueForVariablePath("ssp.object").GetSummary())
        self.assertEqual(1, frame.GetValueForVariablePath("ssp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("ssp.weak_count").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("ssp.foobar").IsValid())
        
        self.assertNotEqual(0, frame.GetValueForVariablePath("iwp.pointer").GetValueAsUnsigned())
        self.assertEqual(123, frame.GetValueForVariablePath("iwp.object").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("iwp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("iwp.weak_count").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("iwp.foobar").IsValid())

        self.assertNotEqual(0, frame.GetValueForVariablePath("swp.pointer").GetValueAsUnsigned())
        self.assertEqual('"foobar"', frame.GetValueForVariablePath("swp.object").GetSummary())
        self.assertEqual(1, frame.GetValueForVariablePath("swp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("swp.weak_count").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("swp.foobar").IsValid())

        self.runCmd("continue")

        frame = self.frame()
        self.assertTrue(frame.IsValid())

        self.expect("frame variable nsp", substrs=['nsp = nullptr'])
        self.expect("frame variable isp", substrs=['isp = nullptr'])
        self.expect("frame variable ssp", substrs=['ssp = nullptr'])
        self.expect("frame variable nwp", substrs=['nwp = nullptr'])
        self.expect("frame variable iwp", substrs=['iwp = nullptr', 'strong=0', 'weak=1'])
        self.expect("frame variable swp", substrs=['swp = nullptr', 'strong=0', 'weak=1'])

        self.assertFalse(frame.GetValueForVariablePath("iwp.object").IsValid())
        self.assertEqual(0, frame.GetValueForVariablePath("iwp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("iwp.weak_count").GetValueAsUnsigned())

        self.assertFalse(frame.GetValueForVariablePath("swp.object").IsValid())
        self.assertEqual(0, frame.GetValueForVariablePath("swp.count").GetValueAsUnsigned())
        self.assertEqual(1, frame.GetValueForVariablePath("swp.weak_count").GetValueAsUnsigned())
