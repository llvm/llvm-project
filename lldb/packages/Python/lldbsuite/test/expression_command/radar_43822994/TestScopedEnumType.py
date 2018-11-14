from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprXValuePrintingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                          '// Set break point at this line.', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        self.expect("expr f == Foo::FooBar",
                substrs=['(bool) $0 = true'])

        value = frame.EvaluateExpression("f == Foo::FooBar")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsUnsigned(), 1)
