from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprBug35310(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_issue35310(self):
        """Test invoking functions with non-standard linkage names.

        The GNU abi_tag extension used by libstdc++ is a common source
        of these, but they could originate from other reasons as well.
        """
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        value = frame.EvaluateExpression("a.test_abi_tag()")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(0), 1)

        value = frame.EvaluateExpression("a.test_asm_name()")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(0), 2)
