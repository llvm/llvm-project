"""
Test that importing the std module on a compile unit
that doesn't use the std module will not break LLDB.

It's not really specified at the moment what kind of
error we should report back to the user in this
situation. Currently Clang will just complain that
the std module doesn't exist or can't be loaded.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class STLTestCase(TestBase):

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

        # Activate importing of std module.
        self.runCmd("settings set target.import-std-module true")

        # Run some commands that should all fail without our std module.
        self.expect("expr std::abs(-42)", error=True)
        self.expect("expr std::div(2, 1).quot", error=True)
        self.expect("expr (std::size_t)33U", error=True)
        self.expect("expr char a = 'b'; char b = 'a'; std::swap(a, b); a",
                    error=True)
