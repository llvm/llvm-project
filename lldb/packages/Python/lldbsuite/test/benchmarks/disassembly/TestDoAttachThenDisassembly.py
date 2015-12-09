"""Test lldb's disassemblt speed.  This bench deliberately attaches to an lldb
inferior and traverses the stack for thread0 to arrive at frame with function
'MainLoop'.  It is important to specify an lldb executable as the inferior."""

from __future__ import print_function



import os, sys
import lldb
from lldbsuite.test.lldbbench import *
from lldbsuite.test import configuration

class AttachThenDisassemblyBench(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        BenchBase.setUp(self)
        self.exe = lldbtest_config.lldbExec
        self.count = configuration.bmIterationCount
        if self.count <= 0:
            self.count = 10

    @benchmarks_test
    @no_debug_info_test
    def test_attach_then_disassembly(self):
        """Attach to a spawned lldb process then run disassembly benchmarks."""
        print()
        self.run_lldb_attach_then_disassembly(self.exe, self.count)
        print("lldb disassembly benchmark:", self.stopwatch)

    def run_lldb_attach_then_disassembly(self, exe, count):
        target = self.dbg.CreateTarget(exe)

        # Spawn a new process and don't display the stdout if not in TraceOn() mode.
        import subprocess
        popen = subprocess.Popen([exe, self.lldbOption],
                                 stdout = open(os.devnull, 'w') if not self.TraceOn() else None)
        if self.TraceOn():
            print("pid of spawned process: %d" % popen.pid)

        # Attach to the launched lldb process.
        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        # Set thread0 as the selected thread, followed by the 'MainLoop' frame
        # as the selected frame.  Then do disassembly on the function.
        thread0 = process.GetThreadAtIndex(0)
        process.SetSelectedThread(thread0)
        i = 0
        found = False
        for f in thread0:
            #print("frame#%d %s" % (i, f.GetFunctionName()))
            if "MainLoop" in f.GetFunctionName():
                found = True
                thread0.SetSelectedFrame(i)
                if self.TraceOn():
                    print("Found frame#%d for function 'MainLoop'" % i)
                break
            i += 1
            
        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            with self.stopwatch:
                # Disassemble the function.
                self.runCmd("disassemble -f")
