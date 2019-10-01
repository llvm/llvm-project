"""
Test basic std::forward_list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicForwardList(TestBase):

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

        self.expect("expr (size_t)std::distance(a.begin(), a.end())", substrs=['(size_t) $0 = 3'])
        self.expect("expr (int)a.front()", substrs=['(int) $1 = 3'])

        self.expect("expr a.sort()")
        self.expect("expr (int)a.front()", substrs=['(int) $2 = 1'])

        self.expect("expr (int)(*a.begin())", substrs=['(int) $3 = 1'])

