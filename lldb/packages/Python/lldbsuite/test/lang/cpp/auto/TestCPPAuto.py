"""
Tests that auto types work
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPAutoTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        compiler="gcc",
        bugnumber="GCC generates incomplete debug info")
    @expectedFailureAll(oslist=['windows'], bugnumber="llvm.org/pr26339")
    def test_with_run_command(self):
        """Test that auto types work in the expression parser"""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        line = line_number('main.cpp', '// break here')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line, num_expected_locations=-1, loc_exact=False)

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect('expr auto f = 123456; f', substrs=['int', '123456'])
        self.expect(
            'expr struct Test { int x; int y; Test() : x(123), y(456) {} }; auto t = Test(); t',
            substrs=[
                'Test',
                '123',
                '456'])
        self.expect(
            'expr auto s = helloworld; s',
            substrs=[
                'string',
                'hello world'])
