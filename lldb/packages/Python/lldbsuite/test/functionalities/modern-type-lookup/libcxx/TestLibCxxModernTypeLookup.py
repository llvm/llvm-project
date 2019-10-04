from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxxModernTypeLookup(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    def test(self):
        self.build()

        # Activate modern-type-lookup.
        self.runCmd("settings set target.experimental.use-modern-type-lookup true")

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        # Test a few simple expressions that should still work with modern-type-lookup.
        self.expect("expr pair", substrs=["(std::", "pair<int, long", "= (first = 1, second = 2)"])
        self.expect("expr foo", substrs=["(std::", "string", "\"bar\""])
        self.expect("expr map", substrs=["(std::", "map", "first = 1, second = 2"])
        self.expect("expr umap", substrs=["(std::", "unordered_map", "first = 1, second = 2"])
