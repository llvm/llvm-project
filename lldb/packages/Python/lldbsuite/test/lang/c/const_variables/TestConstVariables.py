"""Check that compiler-generated constant values work correctly"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ConstVariableTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["freebsd", "linux"],
        compiler="clang", compiler_version=["<", "3.5"])
    @expectedFailureAll(
        oslist=["freebsd", "linux"],
        compiler="clang", compiler_version=["=", "3.7"])
    @expectedFailureAll(
        oslist=["freebsd", "linux"],
        compiler="clang", compiler_version=[">=", "3.8"])
    @expectedFailureAll(oslist=["freebsd", "linux"], compiler="icc")
    @expectedFailureAll(archs=['mips', 'mipsel', 'mips64', 'mips64el'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489: Name lookup not working correctly on Windows")
    def test_and_run_command(self):
        """Test interpreted and JITted expressions on constant values."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = [' resolved, hit count = 1'])

        self.runCmd("next")
        self.runCmd("next")

        # Try frame variable.
        self.expect("frame variable index", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(int32_t) index = 512'])

        # Try an interpreted expression.
        self.expect("expr (index + 512)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['1024'])

        # Try a JITted expression.
        self.expect("expr (int)getpid(); (index - 256)", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['256'])

        self.runCmd("kill")
