"""
Tests that bool types work
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class CPPTestDiamondInheritance(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)
    
    def test_with_run_command(self):
        """Test that virtual base classes work in when SBValue objects are used to explore the variable value"""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.set_breakpoint(line_number('main.cpp', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.cpp', '// breakpoint 2'))
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        frame = thread.GetFrameAtIndex(0)
        j1 = frame.FindVariable("j1")
        j1_Derived1 = j1.GetChildAtIndex(0)
        j1_Derived2 = j1.GetChildAtIndex(1)
        j1_Derived1_VBase = j1_Derived1.GetChildAtIndex(0)
        j1_Derived2_VBase = j1_Derived2.GetChildAtIndex(0)
        j1_Derived1_VBase_m_value = j1_Derived1_VBase.GetChildAtIndex(0)
        j1_Derived2_VBase_m_value = j1_Derived2_VBase.GetChildAtIndex(0)
        self.assertTrue(j1_Derived1_VBase.GetLoadAddress() == j1_Derived2_VBase.GetLoadAddress(), "ensure virtual base class is the same between Derived1 and Derived2")
        self.assertTrue(j1_Derived1_VBase_m_value.GetValueAsUnsigned(1) == j1_Derived2_VBase_m_value.GetValueAsUnsigned(2), "ensure m_value in VBase is the same")
        self.assertTrue(frame.FindVariable("d").GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned(0) == 12345, "ensure Derived2 from j1 is correct");
        thread.StepOver()
        self.assertTrue(frame.FindVariable("d").GetChildAtIndex(0).GetChildAtIndex(0).GetValueAsUnsigned(0) == 12346, "ensure Derived2 from j2 is correct");
    
    def set_breakpoint(self, line):
        # Some compilers (for example GCC 4.4.7 and 4.6.1) emit multiple locations for the statement with the ternary
        # operator in the test program, while others emit only 1.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", line, num_expected_locations=-1, loc_exact=False)
