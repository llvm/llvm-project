"""
Test importing the 'std' C++ module and check if we can handle
prioritizing the conflicting functions from debug info and std
module.

See also import-std-module/basic/TestImportStdModule.py for
the same test on a 'clean' code base without conflicts.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestImportStdModuleConflicts(TestBase):

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
        self.expect("expr std::abs(-42)", substrs=['(int) $0 = 42'])
        self.expect("expr std::div(2, 1).quot", substrs=['(int) $1 = 2'])
        self.expect("expr (std::size_t)33U", substrs=['(size_t) $2 = 33'])
        self.expect("expr char char_a = 'b'; char char_b = 'a'; std::swap(char_a, char_b); char_a",
                    substrs=["(char) $3 = 'a'"])
