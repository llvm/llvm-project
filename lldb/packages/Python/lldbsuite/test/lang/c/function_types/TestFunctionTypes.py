"""Test variable with function ptr type and that break on the function works."""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class FunctionTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def test(self):
        """Test 'callback' has function ptr type, then break on the function."""
        self.build()
        self.runToBreakpoint()

        # Check that the 'callback' variable display properly.
        self.expect("frame variable --show-types callback", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(int (*)(const char *)) callback =')

        # And that we can break on the callback function.
        lldbutil.run_break_set_by_symbol (self, "string_not_empty", num_expected_locations=1, sym_exact=True)
        self.runCmd("continue")

        # Check that we do indeed stop on the string_not_empty function.
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['a.out`string_not_empty',
                       'stop reason = breakpoint'])
    
    @expectedFailureWindows("llvm.org/pr21765")
    def test_pointers(self):
        """Test that a function pointer to 'printf' works and can be called."""
        self.build()
        self.runToBreakpoint()

        self.expect("expr string_not_empty",
                    substrs = ['(int (*)(const char *)) $0 = ', '(a.out`'])

        if self.platformIsDarwin():
            regexps = ['lib.*\.dylib`printf']
        else:
            regexps = ['printf']
        self.expect("expr (int (*)(const char*, ...))printf",
                    substrs = ['(int (*)(const char *, ...)) $1 = '],
                    patterns = regexps)

        self.expect("expr $1(\"Hello world\\n\")",
                    startstr = '(int) $2 = 12')

    def runToBreakpoint(self):
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        
        self.runCmd("run", RUN_SUCCEEDED)
        
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs = ['stopped',
                               'stop reason = breakpoint'])
        
        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs = [' resolved, hit count = 1'])
