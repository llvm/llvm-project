"""
Test more expression command sequences with objective-c.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

@skipUnlessDarwin
class FoundationTestCase2(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break at.
        self.lines = []
        self.lines.append(line_number('main.m', '// Break here for selector: tests'))
        self.lines.append(line_number('main.m', '// Break here for NSArray tests'))
        self.lines.append(line_number('main.m', '// Break here for NSString tests'))
        self.lines.append(line_number('main.m', '// Break here for description test'))
        self.lines.append(line_number('main.m', '// Set break point at this line'))

    def test_more_expr_commands(self):
        """More expression commands for objective-c."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a bunch of breakpoints.
        for line in self.lines:
            lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_Selector:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(sel)",
            substrs = ["(char *)",
                       "length"])

        self.runCmd("process continue")

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.runCmd("process continue")

        # Test_MyString:
        self.runCmd("thread backtrace")
        self.expect("expression (char *)sel_getName(_cmd)",
            substrs = ["(char *)",
                       "description"])

        self.runCmd("process continue")

    def test_NSArray_expr_commands(self):
        """Test expression commands for NSArray."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSArray:
        line = self.lines[1]
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSArray:
        self.runCmd("thread backtrace")
        self.expect("expression (int)[nil_mutable_array count]",
            patterns = ["\(int\) \$.* = 0"])
        self.expect("expression (int)[array1 count]",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)[array2 count]",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)array1.count",
            patterns = ["\(int\) \$.* = 3"])
        self.expect("expression (int)array2.count",
            patterns = ["\(int\) \$.* = 3"])
        self.runCmd("process continue")

    def test_NSString_expr_commands(self):
        """Test expression commands for NSString."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside Test_NSString:
        line = self.lines[2]
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression (int)[str length]",
            patterns = ["\(int\) \$.* ="])
        self.expect("expression (int)[str_id length]",
            patterns = ["\(int\) \$.* ="])
        self.expect("expression [str description]",
            patterns = ["\(id\) \$.* = 0x"])
        self.expect("expression (id)[str_id description]",
            patterns = ["\(id\) \$.* = 0x"])
        self.expect("expression str.length")
        self.expect("expression str.description")
        self.expect('expression str = @"new"')
        self.runCmd("image lookup -t NSString")
        self.expect('expression str = [NSString stringWithCString: "new"]')
        self.runCmd("process continue")

    def test_MyString_dump(self):
        """Test dump of a known Objective-C object by dereferencing it."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        
        self.expect("expression --show-types -- *my",
            patterns = ["\(MyString\) \$.* = ", "\(MyBase\)", "\(NSObject\)", "\(Class\)"])
        self.runCmd("process continue")

    @expectedFailurei386
    def test_NSError_po(self):
        """Test that po of the result of an unknown method doesn't require a cast."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect('po [NSError errorWithDomain:@"Hello" code:35 userInfo:@{@"NSDescription" : @"be completed."}]',
            substrs = ["Error Domain=Hello", "Code=35", "be completed."])
        self.runCmd("process continue")
        
    def test_NSError_p(self):
        """Test that p of the result of an unknown method does require a cast."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        
        line = self.lines[4]

        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("p [NSError thisMethodIsntImplemented:0]",
                    error = True, 
                    patterns = ["no known method", "cast the message send to the method's return type"])
        self.runCmd("process continue")
