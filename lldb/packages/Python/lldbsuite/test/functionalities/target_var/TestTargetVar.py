"""
Test that target var can resolve complex DWARF expressions.
"""

import lldb
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class targetCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIfDarwinEmbedded           # needs x86_64
    @skipIf(debug_info="gmodules")  # not relevant
    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def testTargetVarExpr(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, 'main')
        self.expect("target variable i", substrs=['i', '42'])
