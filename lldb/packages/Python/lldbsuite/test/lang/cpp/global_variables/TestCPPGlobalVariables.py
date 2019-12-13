"""Test that C++ global variables can be inspected by name and also their mangled name."""



from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class GlobalVariablesCppTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = lldb.SBFileSpec('main.cpp')

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    def test(self):
        self.build()

        (target, _, _, _) = lldbutil.run_to_source_breakpoint(self, "// Set break point at this line.", self.source)
        
        # Check that we can access g_file_global_int by its name
        self.expect("target variable g_file_global_int", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['42'])
        self.expect("target variable abc::g_file_global_int", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['42'])
        self.expect("target variable xyz::g_file_global_int", VARIABLES_DISPLAYED_CORRECTLY,
                    error=True, substrs=['can\'t find global variable'])

        # Check that we can access g_file_global_int by its mangled name
        addr = target.EvaluateExpression("&abc::g_file_global_int").GetValueAsUnsigned()
        self.assertTrue(addr != 0)
        mangled = lldb.SBAddress(addr, target).GetSymbol().GetMangledName()
        self.assertTrue(mangled != None)
        gv = target.FindFirstGlobalVariable(mangled)
        self.assertTrue(gv.IsValid())
        self.assertEqual(gv.GetName(), "abc::g_file_global_int")
        self.assertEqual(gv.GetValueAsUnsigned(), 42)
