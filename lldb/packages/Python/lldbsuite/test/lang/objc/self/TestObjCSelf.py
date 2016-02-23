"""
Tests that ObjC member variables are available where they should be.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ObjCSelfTestCase(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)
    
    @skipUnlessDarwin
    def test_with_run_command(self):
        """Test that the appropriate member variables are available when stopped in Objective-C class and instance methods"""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.m', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.m', '// breakpoint 2'))

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect("expression -- m_a = 2",
                    startstr = "(int) $0 = 2")
        
        self.runCmd("process continue")
        
        # This would be disallowed if we enforced const.  But we don't.
        self.expect("expression -- m_a = 2",
                    error=True)
        
        self.expect("expression -- s_a", 
                    startstr = "(int) $1 = 5")

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line (self, "main.m", line, num_expected_locations=1, loc_exact=True)
