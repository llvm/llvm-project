from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class BasicModernTypeLookup(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()

        # Activate modern-type-lookup.
        self.runCmd("settings set target.experimental.use-modern-type-lookup true")

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        # Test a few simple expressions that should still work with modern-type-lookup.
        self.expect("expr 1", substrs=["(int) ", " = 1\n"])
        self.expect("expr f.x", substrs=["(int) ", " = 44\n"])
        self.expect("expr f", substrs=["(Foo) ", "x = 44"])
