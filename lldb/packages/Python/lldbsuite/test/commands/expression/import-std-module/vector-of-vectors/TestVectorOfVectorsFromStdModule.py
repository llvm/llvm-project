"""
Test std::vector functionality when it's contents are vectors.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestVectorOfVectors(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (size_t)a.size()", substrs=['(size_t) $0 = 2'])
        self.expect("expr (int)a.front().front()", substrs=['(int) $1 = 1'])
        self.expect("expr (int)a[1][1]", substrs=['(int) $2 = 2'])
        self.expect("expr (int)a.back().at(0)", substrs=['(int) $3 = 3'])
