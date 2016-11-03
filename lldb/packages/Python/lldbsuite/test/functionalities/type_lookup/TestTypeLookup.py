"""
Test type lookup command.
"""

from __future__ import print_function


import datetime
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeLookupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.mm', '// break here')

    @skipUnlessDarwin
    @skipIf(archs=['i386'])
    def test_type_lookup(self):
        """Test type lookup command."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.mm", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect(
            'type lookup NoSuchType',
            substrs=['@interface'],
            matching=False)
        self.expect('type lookup NSURL', substrs=['NSURL'])
        self.expect('type lookup NSArray', substrs=['NSArray'])
        self.expect('type lookup NSObject', substrs=['NSObject', 'isa'])
        self.expect('type lookup PleaseDontBeARealTypeThatExists', substrs=[
                    "no type was found matching 'PleaseDontBeARealTypeThatExists'"])
        self.expect('type lookup MyCPPClass', substrs=['setF', 'float getF'])
        self.expect('type lookup MyClass', substrs=['setF', 'float getF'])
        self.expect('type lookup MyObjCClass', substrs=['@interface MyObjCClass', 'int x', 'int y'])
