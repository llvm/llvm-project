"""
Test basic std::vector functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicVector(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # FIXME: This should work on more setups, so remove these
    # skipIf's in the future.
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(oslist=no_match(["linux"]))
    @skipIf(debug_info=no_match(["dwarf"]))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect("expr (size_t)a.size()", substrs=['(size_t) $0 = 3'])
        self.expect("expr (int)a.front()", substrs=['(int) $1 = 3'])
        self.expect("expr (int)a[1]", substrs=['(int) $2 = 1'])
        self.expect("expr (int)a.back()", substrs=['(int) $3 = 2'])

        self.expect("expr std::sort(a.begin(), a.end())")
        self.expect("expr (int)a.front()", substrs=['(int) $4 = 1'])
        self.expect("expr (int)a[1]", substrs=['(int) $5 = 2'])
        self.expect("expr (int)a.back()", substrs=['(int) $6 = 3'])

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect("expr (int)a.front()", substrs=['(int) $7 = 3'])
        self.expect("expr (int)a[1]", substrs=['(int) $8 = 2'])
        self.expect("expr (int)a.back()", substrs=['(int) $9 = 1'])

        self.expect("expr (int)(*a.begin())", substrs=['(int) $10 = 3'])
        self.expect("expr (int)(*a.rbegin())", substrs=['(int) $11 = 1'])

        self.expect("expr a.pop_back()")
        self.expect("expr (int)a.back()", substrs=['(int) $12 = 2'])
        self.expect("expr (size_t)a.size()", substrs=['(size_t) $13 = 2'])

        self.expect("expr (int)a.at(0)", substrs=['(int) $14 = 3'])

        self.expect("expr a.push_back(4)")
        self.expect("expr (int)a.back()", substrs=['(int) $15 = 4'])
        self.expect("expr (size_t)a.size()", substrs=['(size_t) $16 = 3'])

        self.expect("expr a.emplace_back(5)")
        self.expect("expr (int)a.back()", substrs=['(int) $17 = 5'])
        self.expect("expr (size_t)a.size()", substrs=['(size_t) $18 = 4'])
