"""
Tests std::stack functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestStack(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        # Test std::stack functionality with a std::deque.
        self.expect("expr s_deque.pop()")
        self.expect("expr s_deque.push({4})")
        self.expect("expr (size_t)s_deque.size()", substrs=['(size_t) $0 = 3'])
        self.expect("expr (int)s_deque.top().i", substrs=['(int) $1 = 4'])
        self.expect("expr s_deque.emplace(5)")
        self.expect("expr (int)s_deque.top().i", substrs=['(int) $2 = 5'])

        # Test std::stack functionality with a std::vector.
        self.expect("expr s_vector.pop()")
        self.expect("expr s_vector.push({4})")
        self.expect("expr (size_t)s_vector.size()", substrs=['(size_t) $3 = 3'])
        self.expect("expr (int)s_vector.top().i", substrs=['(int) $4 = 4'])
        self.expect("expr s_vector.emplace(5)")
        self.expect("expr (int)s_vector.top().i", substrs=['(int) $5 = 5'])

        # Test std::stack functionality with a std::list.
        self.expect("expr s_list.pop()")
        self.expect("expr s_list.push({4})")
        self.expect("expr (size_t)s_list.size()", substrs=['(size_t) $6 = 3'])
        self.expect("expr (int)s_list.top().i", substrs=['(int) $7 = 4'])
        self.expect("expr s_list.emplace(5)")
        self.expect("expr (int)s_list.top().i", substrs=['(int) $8 = 5'])
