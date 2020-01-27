"""
Tests std::queue functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestQueue(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        # Test std::queue functionality with a std::deque.
        self.expect("expr q_deque.pop()")
        self.expect("expr q_deque.push({4})")
        self.expect("expr (size_t)q_deque.size()", substrs=['(size_t) $0 = 1'])
        self.expect("expr (int)q_deque.front().i", substrs=['(int) $1 = 4'])
        self.expect("expr (int)q_deque.back().i", substrs=['(int) $2 = 4'])
        self.expect("expr q_deque.empty()", substrs=['(bool) $3 = false'])
        self.expect("expr q_deque.pop()")
        self.expect("expr q_deque.emplace(5)")
        self.expect("expr (int)q_deque.front().i", substrs=['(int) $4 = 5'])

        # Test std::queue functionality with a std::list.
        self.expect("expr q_list.pop()")
        self.expect("expr q_list.push({4})")
        self.expect("expr (size_t)q_list.size()", substrs=['(size_t) $5 = 1'])
        self.expect("expr (int)q_list.front().i", substrs=['(int) $6 = 4'])
        self.expect("expr (int)q_list.back().i", substrs=['(int) $7 = 4'])
        self.expect("expr q_list.empty()", substrs=['(bool) $8 = false'])
        self.expect("expr q_list.pop()")
        self.expect("expr q_list.emplace(5)")
        self.expect("expr (int)q_list.front().i", substrs=['(int) $9 = 5'])
