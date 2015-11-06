"""
Test that an SBValue can update and format itself as its type changes
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
import os

class TestSBValueUpdates(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)
    
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    @swiftTest
    def test_with_dsym(self):
        """Test that an SBValue can update and format itself as its type changes"""
        self.buildDsym()
        self.do_test()

    @dwarf_test
    @swiftTest
    def test_with_dwarf(self):
        """Test the Any type"""
        self.buildDwarf()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)
    
    def do_test(self):
        """Test that an SBValue can update and format itself as its type changes"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        
        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex('break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.runCmd("run")

        var_x = self.frame().FindVariable("x",lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetValue() == "1")
        
        self.runCmd("continue")
        
        var_x = self.frame().FindVariable("x",lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetSummary() == '"hi"')
        
        self.runCmd("continue")
        
        var_x = self.frame().FindVariable("x",lldb.eDynamicCanRunTarget)
        var_x.SetPreferSyntheticValue(True)

        self.assertTrue(var_x.GetValue() == "5")

       
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
