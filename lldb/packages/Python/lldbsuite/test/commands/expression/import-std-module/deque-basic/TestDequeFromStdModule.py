"""
Test basic std::list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBasicDeque(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        self.expect_expr("(size_t)a.size()", result_type="size_t", result_value="3")
        self.expect_expr("(int)a.front()", result_type="int", result_value="3")
        self.expect_expr("(int)a.back()", result_type="int", result_value="2")

        self.expect("expr std::sort(a.begin(), a.end())")
        self.expect_expr("(int)a.front()", result_type="int", result_value="1")
        self.expect_expr("(int)a.back()", result_type="int", result_value="3")

        self.expect("expr std::reverse(a.begin(), a.end())")
        self.expect_expr("(int)a.front()", result_type="int", result_value="3")
        self.expect_expr("(int)a.back()", result_type="int", result_value="1")

        self.expect_expr("(int)(*a.begin())", result_type="int", result_value="3")
        self.expect_expr("(int)(*a.rbegin())", result_type="int", result_value="1")

