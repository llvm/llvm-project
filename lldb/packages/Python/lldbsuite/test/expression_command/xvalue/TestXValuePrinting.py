from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprXValuePrintingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self, dictionary=None):
        """Printing an xvalue should work."""
        self.build(dictionary=dictionary)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Break here', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        value = frame.EvaluateExpression("foo().data")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(), 1234)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test(self):
        self.do_test()

