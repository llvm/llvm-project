"""
Test 'frame select' command.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestFrameSelect(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    @skipIfWindows
    def test_relative(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -1", substrs=["nested3() at"])

        self.expect("frame select -r -1", error=True, substrs=["Already at the bottom of the stack."])
        self.expect("frame select -r -2147483647", error=True, substrs=["Already at the bottom of the stack."])

        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -2", substrs=["nested3() at"])
        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -2147483647", substrs=["nested3() at"])
        self.expect("frame select -r 1", substrs=["nested2() at"])

        self.expect("frame select -r 100")
        self.expect("frame select -r 1", error=True, substrs=["Already at the top of the stack."])
