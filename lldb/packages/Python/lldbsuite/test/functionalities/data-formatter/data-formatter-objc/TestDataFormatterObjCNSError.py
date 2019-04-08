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


class ObjCDataFormatterNSError(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nserror_with_run_command(self):
        """Test formatters for NSError."""
        self.appkit_tester_impl(self.nserror_data_formatter_commands)

    def nserror_data_formatter_commands(self):
        self.expect(
            'frame variable nserror', substrs=['domain: @"Foobar" - code: 12'])

        self.expect(
            'frame variable nserrorptr',
            substrs=['domain: @"Foobar" - code: 12'])

        self.expect(
            'frame variable nserror->_userInfo', substrs=['2 key/value pairs'])

        self.expect(
            'frame variable nserror->_userInfo --ptr-depth 1 -d run-target',
            substrs=['@"a"', '@"b"', "1", "2"])
