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


class ObjCDataFormatterNSData(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nsdata_with_run_command(self):
        """Test formatters for  NSData."""
        self.appkit_tester_impl(self.nsdata_data_formatter_commands)

    def nsdata_data_formatter_commands(self):
        self.expect(
            'frame variable immutableData mutableData data_ref mutable_data_ref mutable_string_ref concreteData concreteMutableData',
            substrs=[
                '(NSData *) immutableData = ', ' 4 bytes',
                '(NSData *) mutableData = ', ' 14 bytes',
                '(CFDataRef) data_ref = ', '@"5 bytes"',
                '(CFMutableDataRef) mutable_data_ref = ', '@"5 bytes"',
                '(CFMutableStringRef) mutable_string_ref = ',
                ' @"Wish ya knew"', '(NSData *) concreteData = ',
                ' 100000 bytes', '(NSMutableData *) concreteMutableData = ',
                ' 100000 bytes'
            ])
