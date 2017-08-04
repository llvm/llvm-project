"""
Test that LLDB can emit JIT objects when the appropriate setting is enabled
"""

from __future__ import print_function

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

def enumerateJITFiles():
    return [f for f in os.listdir(os.getcwd()) if f.startswith("jit")]
    
def countJITFiles():
    return len(enumerateJITFiles())

def cleanJITFiles():
    for j in enumerateJITFiles():
        os.remove(j)
    return

class SaveJITObjectsTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"])
    def test_save_jit_objects(self):
        self.build()
        src_file = "main.c"
        src_file_spec = lldb.SBFileSpec(src_file)
  
        exe_path = os.path.join(os.getcwd(), "a.out")
        target = self.dbg.CreateTarget(exe_path)

        breakpoint = target.BreakpointCreateBySourceRegex(
            "break", src_file_spec)

        process = target.LaunchSimple(None, None,
                                      self.get_process_working_directory())

        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

        cleanJITFiles()
        frame.EvaluateExpression("(void*)malloc(0x1)")
        self.assertTrue(countJITFiles() == 0,
                        "No files emitted with save-jit-objects=false")

        self.runCmd("settings set target.save-jit-objects true")
        frame.EvaluateExpression("(void*)malloc(0x1)")
        jit_files_count = countJITFiles()
        cleanJITFiles()
        self.assertTrue(jit_files_count != 0,
                        "At least one file emitted with save-jit-objects=true")

        process.Kill()
