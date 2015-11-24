"""Look up type information for typedefs of same name at different lexical scope and check for correct display."""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class TypedefTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    @expectedFailureClang("llvm.org/pr19238")
    def test_with_dsym(self):
        """Test 'image lookup -t a' and check for correct display at different scopes."""
        self.buildDsym()
        self.image_lookup_for_multiple_typedefs()

    @dwarf_test
    @expectedFailureClang("llvm.org/pr19238")
    @expectedFailureFreeBSD("llvm.org/pr25626 expectedFailureClang fails on FreeBSD")
    def test_with_dwarf(self):
        """Test 'image lookup -t a' and check for correct display at different scopes."""
        self.buildDwarf()
        self.image_lookup_for_multiple_typedefs()

    def image_lookup_for_multiple_typedefs(self):
        """Test 'image lookup -t a' at different scopes and check for correct display."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        typearray = ("float", "float", "char", "double *", "float", "int", "double", "float", "float")
        arraylen = len(typearray)+1
        for i in range(1,arraylen):
            loc_line = line_number('main.c', '// Set break point ' + str(i) + '.')
            lldbutil.run_break_set_by_file_and_line (self, "main.c",loc_line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        for t in typearray:
            self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])
            self.expect("image lookup -t a", DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['name = "' + t + '"'])
            self.runCmd("continue")
