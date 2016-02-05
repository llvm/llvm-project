"""
Test using LLDB data formatters with frozen objects coming from the expression parser.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprFormattersTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp',
                                '// Stop here')

    @skipIfFreeBSD # llvm.org/pr24691 skipping to avoid crashing the test runner
    @expectedFailureFreeBSD('llvm.org/pr19011') # Newer Clang omits C1 complete object constructor
    @expectedFailureFreeBSD('llvm.org/pr24691') # we hit an assertion in clang
    @expectedFailureWindows("llvm.org/pr21765")
    @skipIfTargetAndroid() # skipping to avoid crashing the test runner
    @expectedFailureAndroid('llvm.org/pr24691') # we hit an assertion in clang
    def test(self):
        """Test expr + formatters for good interoperability."""
        self.build()

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synthetic clear', check=False)
        
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """Test expr + formatters for good interoperability."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.runCmd("command script import formatters.py")
        self.runCmd("command script import foosynth.py")
        
        if self.TraceOn():
            self.runCmd("frame variable foo1 --show-types")
            self.runCmd("frame variable foo1.b --show-types")
            self.runCmd("frame variable foo1.b.b_ref --show-types")

        self.expect("expression --show-types -- *(new foo(47))",
            substrs = ['(int) a = 47', '(bar) b = {', '(int) i = 94', '(baz) b = {', '(int) k = 99'])

        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect("expression new int(12)",
            substrs = ['(int *) $', ' = 0x'])

        self.runCmd("type summary add -s \"${var%pointer} -> ${*var%decimal}\" \"int *\"")

        self.expect("expression new int(12)",
            substrs = ['(int *) $', '= 0x', ' -> 12'])

        self.expect("expression foo1.a_ptr",
            substrs = ['(int *) $', '= 0x', ' -> 13'])

        self.expect("expression foo1",
            substrs = ['(foo) $', ' a = 12', 'a_ptr = ', ' -> 13','i = 24','i_ptr = ', ' -> 25'])

        self.expect("expression --ptr-depth=1 -- new foo(47)",
            substrs = ['(foo *) $', 'a = 47','a_ptr = ', ' -> 48','i = 94','i_ptr = ', ' -> 95'])

        self.expect("expression foo2",
            substrs = ['(foo) $', 'a = 121','a_ptr = ', ' -> 122','i = 242','i_ptr = ', ' -> 243'])

        object_name = self.res.GetOutput()
        object_name = object_name[7:]
        object_name = object_name[0:object_name.find(' =')]

        self.expect("frame variable foo2",
                    substrs = ['(foo)', 'foo2', 'a = 121','a_ptr = ', ' -> 122','i = 242','i_ptr = ', ' -> 243'])
        
        self.expect("expression $" + object_name,
            substrs = ['(foo) $', 'a = 121','a_ptr = ', ' -> 122','i = 242','i_ptr = ', ' -> 243', 'h = 245','k = 247'])

        self.runCmd("type summary delete foo")
        self.runCmd("type synthetic add --python-class foosynth.FooSyntheticProvider foo")

        self.expect("expression --show-types -- $" + object_name,
            substrs = ['(foo) $', ' = {', '(int) *i_ptr = 243'])

        self.runCmd("n")
        self.runCmd("n")

        self.runCmd("type synthetic delete foo")
        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect("expression foo2",
            substrs = ['(foo) $', 'a = 7777','a_ptr = ', ' -> 122','i = 242','i_ptr = ', ' -> 8888'])

        self.expect("expression $" + object_name + '.a',
            substrs = ['7777'])

        self.expect("expression *$" + object_name + '.b.i_ptr',
            substrs = ['8888'])

        self.expect("expression $" + object_name,
            substrs = ['(foo) $', 'a = 121', 'a_ptr = ', ' -> 122', 'i = 242', 'i_ptr = ', ' -> 8888', 'h = 245','k = 247'])

        self.runCmd("type summary delete foo")
        self.runCmd("type synthetic add --python-class foosynth.FooSyntheticProvider foo")

        self.expect("expression --show-types -- $" + object_name,
            substrs = ['(foo) $', ' = {', '(int) *i_ptr = 8888'])

        self.runCmd("n")

        self.runCmd("type synthetic delete foo")
        self.runCmd("type summary add -F formatters.foo_SummaryProvider foo")

        self.expect("expression $" + object_name,
                    substrs = ['(foo) $', 'a = 121','a_ptr = ', ' -> 122','i = 242', 'i_ptr = ', ' -> 8888','k = 247'])

        process = self.dbg.GetSelectedTarget().GetProcess()
        thread = process.GetThreadAtIndex(0)
        frame = thread.GetSelectedFrame()

        frozen = frame.EvaluateExpression("$" + object_name + ".a_ptr")

        a_data = frozen.GetPointeeData()

        error = lldb.SBError()
        self.assertTrue(a_data.GetUnsignedInt32(error, 0) == 122, '*a_ptr = 122')

        self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers",
                    substrs = ['1','2','3','4','5'])

        self.expect("expression numbers",
                    substrs = ['1','2','3','4','5'])

        frozen = frame.EvaluateExpression("&numbers")

        a_data = frozen.GetPointeeData(0, 1)

        self.assertTrue(a_data.GetUnsignedInt32(error, 0) == 1, 'numbers[0] == 1')
        self.assertTrue(a_data.GetUnsignedInt32(error, 4) == 2, 'numbers[1] == 2')
        self.assertTrue(a_data.GetUnsignedInt32(error, 8) == 3, 'numbers[2] == 3')
        self.assertTrue(a_data.GetUnsignedInt32(error, 12) == 4, 'numbers[3] == 4')
        self.assertTrue(a_data.GetUnsignedInt32(error, 16) == 5, 'numbers[4] == 5')

        frozen = frame.EvaluateExpression("numbers")

        a_data = frozen.GetData()

        self.assertTrue(a_data.GetUnsignedInt32(error, 0) == 1, 'numbers[0] == 1')
        self.assertTrue(a_data.GetUnsignedInt32(error, 4) == 2, 'numbers[1] == 2')
        self.assertTrue(a_data.GetUnsignedInt32(error, 8) == 3, 'numbers[2] == 3')
        self.assertTrue(a_data.GetUnsignedInt32(error, 12) == 4, 'numbers[3] == 4')
        self.assertTrue(a_data.GetUnsignedInt32(error, 16) == 5, 'numbers[4] == 5')
