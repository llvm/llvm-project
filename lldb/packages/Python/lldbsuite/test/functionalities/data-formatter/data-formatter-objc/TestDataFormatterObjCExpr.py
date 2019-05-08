# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterExpr(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_expr_with_run_command(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set break point at this line.',
            lldb.SBFileSpec('main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # check that the formatters are able to deal safely and correctly
        # with ValueObjects that the expression parser returns
        self.expect(
            'expression ((id)@"Hello for long enough to avoid short string types")',
            matching=False,
            substrs=['Hello for long enough to avoid short string types'])

        self.expect(
            'expression -d run -- ((id)@"Hello for long enough to avoid short string types")',
            substrs=['Hello for long enough to avoid short string types'])

        self.expect('expr -d run -- label1', substrs=['Process Name'])

        self.expect(
            'expr -d run -- @"Hello for long enough to avoid short string types"',
            substrs=['Hello for long enough to avoid short string types'])

        self.expect(
            'expr -d run --object-description -- @"Hello for long enough to avoid short string types"',
            substrs=['Hello for long enough to avoid short string types'])
        self.expect(
            'expr -d run --object-description -- @"Hello"',
            matching=False,
            substrs=['@"Hello" Hello'])
