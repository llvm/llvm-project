"""
Test basic std::weak_ptr functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestSharedPtr(TestBase):

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

        self.expect("expr (int)*w.lock()", substrs=['(int) $0 = 3'])
        self.expect("expr (int)(*w.lock() = 5)", substrs=['(int) $1 = 5'])
        self.expect("expr (int)*w.lock()", substrs=['(int) $2 = 5'])
        self.expect("expr w.use_count()", substrs=['(long) $3 = 1'])
        self.expect("expr w.reset()")
        self.expect("expr w.use_count()", substrs=['(long) $4 = 0'])

