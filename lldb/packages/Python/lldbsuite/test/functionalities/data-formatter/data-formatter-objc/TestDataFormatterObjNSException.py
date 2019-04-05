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


class ObjCDataFormatterNSException(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nsexception_with_run_command(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.nsexception_data_formatter_commands)

    def nsexception_data_formatter_commands(self):
        self.expect(
            'frame variable except0 except1 except2 except3',
            substrs=[
                '(NSException *) except0 = ',
                'name: @"TheGuyWhoHasNoName" - reason: @"cuz it\'s funny"',
                '(NSException *) except1 = ',
                'name: @"TheGuyWhoHasNoName~1" - reason: @"cuz it\'s funny"',
                '(NSException *) except2 = ',
                'name: @"TheGuyWhoHasNoName`2" - reason: @"cuz it\'s funny"',
                '(NSException *) except3 = ',
                'name: @"TheGuyWhoHasNoName/3" - reason: @"cuz it\'s funny"'
            ])
