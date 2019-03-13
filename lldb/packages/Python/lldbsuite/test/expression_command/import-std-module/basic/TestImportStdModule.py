"""
Test importing the 'std' C++ module and evaluate expressions with it.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ImportStdModule(TestBase):

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
        # Calling some normal std functions that return non-template types.
        self.expect("expr std::abs(-42)", substrs=['(int) $0 = 42'])
        self.expect("expr std::div(2, 1).quot", substrs=['(int) $1 = 2'])
        # Using types from std.
        self.expect("expr (std::size_t)33U", substrs=['(size_t) $2 = 33'])
        # Calling templated functions that return non-template types.
        self.expect("expr char a = 'b'; char b = 'a'; std::swap(a, b); a",
                    substrs=["(char) $3 = 'a'"])

    # FIXME: This should work on more setups, so remove these
    # skipIf's in the future.
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(oslist=no_match(["linux"]))
    @skipIf(debug_info=no_match(["dwarf"]))
    def test_non_cpp_language(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        # Activate importing of std module.
        self.runCmd("settings set target.import-std-module true")
        # These languages don't support C++ modules, so they shouldn't
        # be able to evaluate the expression.
        self.expect("expr -l C -- std::abs(-42)", error=True)
        self.expect("expr -l C99 -- std::abs(-42)", error=True)
        self.expect("expr -l C11 -- std::abs(-42)", error=True)
        self.expect("expr -l ObjC -- std::abs(-42)", error=True)
