"""
Test that we can have persistent pointer variables
"""

from __future__ import print_function


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class PersistentPtrUpdateTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def test(self):
        """Test that we can have persistent pointer variables"""
        self.build()

        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd('break set -p here')

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expr void* $foo = 0")

        self.runCmd("continue")

        self.expect("expr $foo", substrs=['$foo', '0x0'])
