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


class StdTupleDataFormatterTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD
    @skipIfWindows  # libstdcpp not ported to Windows
    @skipIfDarwin  # doesn't compile on Darwin
    @skipIfwatchOS  # libstdcpp not ported to watchos
    def test_with_run_command(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(
            self, "Set break point at this line.")
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        frame = self.frame()
        self.assertTrue(frame.IsValid())

        self.expect("frame variable ti", substrs=['[0] = 1'])
        self.expect("frame variable ts", substrs=['[0] = "foobar"'])
        self.expect("frame variable tt", substrs=['[0] = 1', '[1] = "baz"', '[2] = 2'])

        self.assertEqual(1, frame.GetValueForVariablePath("ti[0]").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("ti[1]").IsValid())

        self.assertEqual('"foobar"', frame.GetValueForVariablePath("ts[0]").GetSummary())
        self.assertFalse(frame.GetValueForVariablePath("ts[1]").IsValid())
        
        self.assertEqual(1, frame.GetValueForVariablePath("tt[0]").GetValueAsUnsigned())
        self.assertEqual('"baz"', frame.GetValueForVariablePath("tt[1]").GetSummary())
        self.assertEqual(2, frame.GetValueForVariablePath("tt[2]").GetValueAsUnsigned())
        self.assertFalse(frame.GetValueForVariablePath("tt[3]").IsValid())
