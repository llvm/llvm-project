"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxMultiSetDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        ns = 'ndk' if lldbplatformutil.target_is_android() else ''
        self.namespace = 'std::__' + ns + '1'

    def getVariableType(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var.GetType().GetCanonicalType().GetName()

    def check_ii(self, var_name):
        """ This checks the value of the bitset stored in ii at the call to by_ref_and_ptr.
            We use this to make sure we get the same values for ii when we look at the object
            directly, and when we look at a reference to the object. """
        self.expect(
            "frame variable " + var_name,
            substrs=["size=7",
                     "[2] = 2",
                     "[3] = 3",
                     "[6] = 6"])
        self.expect("frame variable " + var_name + "[2]", substrs=[" = 2"])
        self.expect(
            "p " + var_name,
            substrs=[
                "size=7",
                "[2] = 2",
                "[3] = 3",
                "[6] = 6"])

    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp", False))

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ii_type = self.getVariableType("ii")
        self.assertTrue(ii_type.startswith(self.namespace + "::multiset"),
                        "Type: " + ii_type)

        self.expect("frame variable ii", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ii",
            substrs=[
                "size=6",
                "[0] = 0",
                "[1] = 1",
                "[2] = 2",
                "[3] = 3",
                "[4] = 4",
                "[5] = 5"])
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check_ii("ii")

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ii", substrs=["size=0", "{}"])
        ss_type = self.getVariableType("ss")
        self.assertTrue(ss_type.startswith(self.namespace + "::multiset"),
                        "Type: " + ss_type)
        self.expect("frame variable ss", substrs=["size=0", "{}"])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=[
                "size=2",
                '[0] = "a"',
                '[1] = "a very long string is right here"'])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=[
                "size=4",
                '[2] = "b"',
                '[3] = "c"',
                '[0] = "a"',
                '[1] = "a very long string is right here"'])
        self.expect(
            "p ss",
            substrs=[
                "size=4",
                '[2] = "b"',
                '[3] = "c"',
                '[0] = "a"',
                '[1] = "a very long string is right here"'])
        self.expect("frame variable ss[2]", substrs=[' = "b"'])
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable ss",
            substrs=[
                "size=3",
                '[0] = "a"',
                '[1] = "a very long string is right here"',
                '[2] = "c"'])

    @add_test_categories(["libc++"])
    def test_ref_and_ptr(self):
        """Test that the data formatters work on ref and ptr."""
        self.build()
        (self.target, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to check by ref and ptr.",
            lldb.SBFileSpec("main.cpp", False))
        # The reference should print just like the value:
        self.check_ii("ref")

        self.expect("frame variable ptr",
                    substrs=["ptr =", "size=7"])
        self.expect("expr ptr",
                    substrs=["size=7"])
