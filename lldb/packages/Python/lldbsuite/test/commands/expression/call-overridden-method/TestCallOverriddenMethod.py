"""
Test calling an overriden method.

Note:
  This verifies that LLDB is correctly building the method overrides table.
  If this table is not built correctly then calls to overridden methods in
  derived classes may generate references to non-existant vtable entries,
  as the compiler treats the overridden method as a totally new virtual
  method definition.
  <rdar://problem/14205774>

"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprCommandCallOverriddenMethod(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp', '// Set breakpoint here')

    def test_call_on_base(self):
        """Test calls to overridden methods in derived classes."""
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test call to method in base class (this should always work as the base
        # class method is never an override).
        self.expect("expr b->foo()", substrs=["2"])

        # Test calling the base class.
        self.expect("expr realbase.foo()", substrs=["1"])

    @skipIfLinux # Returns wrong result code on some platforms.
    def test_call_on_derived(self):
        """Test calls to overridden methods in derived classes."""
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test call to overridden method in derived class (this will fail if the
        # overrides table is not correctly set up, as Derived::foo will be assigned
        # a vtable entry that does not exist in the compiled program).
        self.expect("expr d.foo()", substrs=["2"])

    @skipIf(oslist=["linux"], archs=["aarch64"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr43707")
    def test_call_on_temporary(self):
        """Test calls to overridden methods in derived classes."""
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test with locally constructed instances.
        self.expect("expr Base().foo()", substrs=["1"])
        self.expect("expr Derived().foo()", substrs=["2"])
