"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class TypeSummaryListArgumentTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @no_debug_info_test
    def test_type_summary_list_with_arg(self):
        """Test that the 'type summary list' command handles command line arguments properly"""
        self.expect('type summary list Foo', substrs=['Category: default', 'Category: system'])
        self.expect('type summary list char', substrs=['char *', 'unsigned char'])
