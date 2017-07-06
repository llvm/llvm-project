from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCharTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self, dictionary=None):
        """These basic expression commands should work as expected."""
        self.build(dictionary=dictionary)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, 
                                          '// Break here', self.main_source_spec)
        frame = thread.GetFrameAtIndex(0)

        value = frame.EvaluateExpression("foo(c)")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(0), 1)

        value = frame.EvaluateExpression("foo(sc)")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(0), 2)

        value = frame.EvaluateExpression("foo(uc)")
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertEqual(value.GetValueAsSigned(0), 3)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_default_char(self):
        self.do_test()

    @expectedFailureAll(
        archs=[
            "arm",
            "aarch64",
            "s390x"],
        bugnumber="llvm.org/pr23069")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_signed_char(self):
        self.do_test(dictionary={'CFLAGS_EXTRAS': '-fsigned-char'})

    @expectedFailureAll(
        archs=[
            "i[3-6]86",
            "x86_64"],
        bugnumber="llvm.org/pr23069, <rdar://problem/28721938>")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    @expectedFailureAll(triple='mips*', bugnumber="llvm.org/pr23069")
    def test_unsigned_char(self):
        self.do_test(dictionary={'CFLAGS_EXTRAS': '-funsigned-char'})
