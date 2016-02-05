"""
Test calling std::String member functions.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprCommandCallFunctionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')

    @expectedFailureIcc # llvm.org/pr14437, fails with ICC 13.1
    @expectedFailureFreeBSD('llvm.org/pr17807') # Fails on FreeBSD buildbot
    @expectedFailureWindows("llvm.org/pr21765")
    def test_with(self):
        """Test calling std::String member function."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        # Some versions of GCC encode two locations for the 'return' statement in main.cpp
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("print str",
            substrs = ['Hello world'])

        # Calling this function now succeeds, but we follow the typedef return type through to
        # const char *, and thus don't invoke the Summary formatter.
        self.expect("print str.c_str()",
            substrs = ['Hello world'])
