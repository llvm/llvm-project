"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterLibcxxForwardList(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number('main.cpp', '// break here')
        ns = 'ndk' if lldbplatformutil.target_is_android() else ''
        self.namespace = 'std::__' + ns + '1'

    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::forward_list is displayed correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        forward_list = self.namespace + '::forward_list'
        self.expect("frame variable empty",
                    substrs=[forward_list,
                             'size=0',
                             '{}'])

        self.expect("frame variable one_elt",
                    substrs=[forward_list,
                             'size=1',
                             '{',
                             '[0] = 47',
                             '}'])

        self.expect("frame variable five_elts",
                    substrs=[forward_list,
                             'size=5',
                             '{',
                             '[0] = 1',
                             '[1] = 22',
                             '[2] = 333',
                             '[3] = 4444',
                             '[4] = 55555',
                             '}'])
