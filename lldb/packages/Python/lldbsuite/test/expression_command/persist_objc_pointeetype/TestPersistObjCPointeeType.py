"""
Test that we can p *objcObject
"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class PersistObjCPointeeType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.m','// break here')

    @skipUnlessDarwin
    @expectedFailureAll(
        bugnumber='http://llvm.org/pr23504',
        oslist=['macosx'], compiler='clang', compiler_version=['<', '7.0.0'])
    def test_with(self):
        """Test that we can p *objcObject"""
        self.build()

        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.expect("p *self", substrs=['_sc_name = nil',
        '_sc_name2 = nil',
        '_sc_name3 = nil',
        '_sc_name4 = nil',
        '_sc_name5 = nil',
        '_sc_name6 = nil',
        '_sc_name7 = nil',
        '_sc_name8 = nil'])
