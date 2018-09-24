import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftTestGeneralizedAccessors(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_generalized_accessors(self):
        self.build()
        self.runtest_accessors()

    def setUp(self):
        TestBase.setUp(self)

    def runtest_accessors(self):
        (target, process, thread, breakpoint) = lldbutil.run_to_source_breakpoint(self, 
                "break here", lldb.SBFileSpec("main.swift"))

        frame = thread.GetFrameAtIndex(0)
        oddnumbers = frame.FindVariable("d")
        self.expect("frame variable -d run -- d",
                    substrs=['key = \"odd\"',
                             'value = 5 values {',
                             '[0] = 1',
                             '[1] = 3',
                             '[2] = 5',
                             '[3] = 7',
                             '[4] = 9'])
        process.Continue()
        self.expect("frame variable -d run -- d",
                    substrs=['key = \"odd\"',
                             'value = 5 values {',
                             '[0] = 2',
                             '[1] = 3',
                             '[2] = 5',
                             '[3] = 7',
                             '[4] = 9'])
        process.Continue()
        self.expect("frame variable -d run -- d",
                    substrs=['key = \"odd\"',
                             'value = 5 values {',
                             '[0] = 2',
                             '[1] = 6',
                             '[2] = 5',
                             '[3] = 7',
                             '[4] = 9'])
        process.Continue()
        self.expect("frame variable -d run -- d",
                    substrs=['key = \"odd\"',
                             'value = 5 values {',
                             '[0] = 2',
                             '[1] = 6',
                             '[2] = 10',
                             '[3] = 7',
                             '[4] = 9'])
        process.Continue()
        self.expect("frame variable -d run -- d",
                    substrs=['key = \"odd\"',
                             'value = 5 values {',
                             '[0] = 2',
                             '[1] = 6',
                             '[2] = 10',
                             '[3] = 14',
                             '[4] = 9'])
