"""
Test that we can compile expressions referring to
absent weak symbols from a dylib.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test import decorators
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestWeakSymbolsInExpressions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @decorators.skipUnlessDarwin
    def test_weak_symbol_in_expr(self):
        """Tests that we can refer to weak symbols in expressions."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def run_weak_var_check (self, weak_varname, present):
        # The expression will modify present_weak_int to signify which branch
        # was taken.  Set it to so we don't get confused by a previous run.
        value = self.target.FindFirstGlobalVariable("present_weak_int")
        value.SetValueFromCString("0")
        if present:
            correct_value = 10
        else:
            correct_value = 20
            
        # Note, I'm adding the "; 10" at the end of the expression to work around
        # the bug that expressions with no result currently return False for Success()...
        expr = "if (&" + weak_varname + " != NULL) { present_weak_int = 10; } else { present_weak_int = 20;}; 10"
        result = self.frame.EvaluateExpression(expr)
        self.assertTrue(result.GetError().Success(), "absent_weak_int expr failed: %s"%(result.GetError().GetCString()))
        self.assertEqual(value.GetValueAsSigned(), correct_value, "Didn't change present_weak_int correctly.")
        
    def do_test(self):
        hidden_dir = os.path.join(self.getBuildDir(), "hidden")
        
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.getBuildDir())
        # We have to point to the hidden directory to pick up the
        # version of the dylib without the weak symbols:
        env_expr = self.platformContext.shlib_environment_var + "=" + hidden_dir
        launch_info.SetEnvironmentEntries([env_expr], True)
        
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here", self.main_source_file,
                                                                            launch_info = launch_info)
        # First we have to import the Dylib module so we get the type info
        # for the weak symbol.  We need to add the source dir to the module
        # search paths, and then run @import to introduce it into the expression
        # context:
        self.dbg.HandleCommand("settings set target.clang-module-search-paths " + self.getSourceDir())
        
        self.frame = thread.frames[0]
        self.assertTrue(self.frame.IsValid(), "Got a good frame")
        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeObjC)
        result = self.frame.EvaluateExpression("@import Dylib", options)

        # Now run an expression that references an absent weak symbol:
        self.run_weak_var_check("absent_weak_int", False)
        self.run_weak_var_check("absent_weak_function", False)
        
        # Make sure we can do the same thing with present weak symbols
        self.run_weak_var_check("present_weak_int", True)
        self.run_weak_var_check("present_weak_function", True)
