"""
Test that Objective-C methods from the runtime work correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class RuntimeTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["macosx"],
        debug_info="gmodules",
        bugnumber="llvm.org/pr27862")
    def test_break(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        if self.getArchitecture() != 'x86_64':
            self.skipTest("This only applies to the v2 runtime")

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at -[MyString description].
        lldbutil.run_break_set_by_symbol(
            self,
            '-[MyString description]',
            num_expected_locations=1,
            sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The backtrace should show we stop at -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
                    substrs=["a.out`-[MyString description]"])

        # Use runtime information about NSString.

        # The length property should be usable.
        self.expect("expression str.length", VARIABLES_DISPLAYED_CORRECTLY,
                    patterns=[r"(\(unsigned long long\))|\(NSUInteger\)"])

        # Static methods on NSString should work.
        self.expect(
            "expr [NSString stringWithCString:\"foo\" encoding:1]",
            VALID_TYPE,
            substrs=[
                "(id)",
                "$1"])

        self.expect("po $1", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["foo"])
