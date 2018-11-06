import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators
import lldbsuite.test.lldbutil as lldbutil


class TestVLA(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipIf(compiler="clang", compiler_version=['<', '8.0'])
    def test_vla(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.c'))

        def test(a, array):
            for i in range(a):
                self.expect("fr v vla[%d]"%i, substrs=["int", "%d"%(a-i)])
                self.expect("expr vla[%d]"%i, substrs=["int", "%d"%(a-i)])
            self.expect("frame var vla", substrs=array)
            self.expect("expr      vla", error=True, substrs=["incomplete"])

        test(2, ["int []", "[0] = 2, [1] = 1"])
        process.Continue()
        test(4, ["int []", "[0] = 4, [1] = 3, [2] = 2, [3] = 1"])

