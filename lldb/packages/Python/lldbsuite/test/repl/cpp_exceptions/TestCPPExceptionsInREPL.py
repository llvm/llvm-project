"""
Make sure we don't trap exceptions with SetREPLEnabled(true) or "lldb --repl"
"""

from __future__ import print_function


import os
import time
import re
import subprocess
import lldb
import swift
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators


class TestREPLExceptions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the 
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @decorators.skipUnlessDarwin
    def test_set_repl_mode_exceptions(self):
        """ Test that SetREPLMode turns off trapping exceptions."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.swift")
        self.do_repl_mode_test()

    @decorators.skipUnlessDarwin
    def test_repl_exceptions(self):
        """ Test the lldb --repl turns off trapping exceptions."""
        self.build()
        self.do_repl_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_repl_test(self):
        sdk_root = swift.getSwiftSDKRoot()
        build_dir = self.getBuildDir()
        repl_args = [lldbtest_config.lldbExec, "-x", "--repl=-enable-objc-interop -sdk %s -L%s -I%s"%(sdk_root, build_dir, build_dir)]
        repl_proc = subprocess.Popen(repl_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=build_dir)
        (stdoutdata, stderrdata) = repl_proc.communicate(input="import Wrapper\ncall_cpp()\n:quit")
        self.assertTrue("I called it successfully" in stdoutdata, "Didn't call call_cpp successfully: out: \n%s\nerr: %s"%(stdoutdata, stderrdata))
        
    def do_repl_mode_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file) 

        frame = thread.GetFrameAtIndex(0)
        options = lldb.SBExpressionOptions()
        options.SetREPLMode(True)
        val = frame.EvaluateExpression("call_cpp(); 5", options)
        self.assertTrue(val.GetError().Success(), 
                        "Got an error evaluating expression: %s."%(val.GetError().GetCString()))
        self.assertEqual(val.GetValueAsUnsigned(), 5,"The expression didn't return the correct result")
