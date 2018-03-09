"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterLibcxxTuple(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number('main.cpp', '// break here')
        ns = 'ndk' if lldbplatformutil.target_is_android() else ''
        self.namespace = 'std::__' + ns + '1'

    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::tuple is displayed correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        tuple_name = self.namespace + '::tuple'
        self.expect("frame variable empty",
                    substrs=[tuple_name,
                             'size=0',
                             '{}'])

        self.expect("frame variable one_elt",
                    substrs=[tuple_name,
                             'size=1',
                             '{',
                             '[0] = 47',
                             '}'])

        self.expect("frame variable three_elts",
                    substrs=[tuple_name,
                             'size=3',
                             '{',
                             '[0] = 1',
                             '[1] = 47',
                             '[2] = "foo"',
                             '}'])
