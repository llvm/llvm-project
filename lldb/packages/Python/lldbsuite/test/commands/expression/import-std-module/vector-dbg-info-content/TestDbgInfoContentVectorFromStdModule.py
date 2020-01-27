"""
Test basic std::vector functionality but with a declaration from
the debug info (the Foo struct) as content.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestDbgInfoContentVector(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (size_t)a.size()", substrs=['(size_t) $0 = 3'])
        self.expect("expr (int)a.front().a", substrs=['(int) $1 = 3'])
        self.expect("expr (int)a[1].a", substrs=['(int) $2 = 1'])
        self.expect("expr (int)a.back().a", substrs=['(int) $3 = 2'])

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect("expr (int)a.front().a", substrs=['(int) $4 = 2'])

        self.expect("expr (int)(a.begin()->a)", substrs=['(int) $5 = 2'])
        self.expect("expr (int)(a.rbegin()->a)", substrs=['(int) $6 = 3'])

        self.expect("expr a.pop_back()")
        self.expect("expr (int)a.back().a", substrs=['(int) $7 = 1'])
        self.expect("expr (size_t)a.size()", substrs=['(size_t) $8 = 2'])

        self.expect("expr (int)a.at(0).a", substrs=['(int) $9 = 2'])

        self.expect("expr a.push_back({4})")
        self.expect("expr (int)a.back().a", substrs=['(int) $10 = 4'])
        self.expect("expr (size_t)a.size()", substrs=['(size_t) $11 = 3'])
