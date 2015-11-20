"""
Test that ASan memory history provider returns correct stack traces
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class AsanTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # The default compiler ("clang") may not support Address Sanitizer or it
    # may not have the debugging API which was recently added, so we're calling
    # self.useBuiltClang() to use clang from the llvm-build directory instead

    @expectedFailureLinux # non-core functionality, need to reenable and fix later (DES 2014.11.07)
    @skipIfFreeBSD # llvm.org/pr21136 runtimes not yet available by default
    @skipIfRemote
    @skipUnlessCompilerRt
    @expectedFailureDarwin
    def test (self):
        compiler = self.findBuiltClang ()
        self.build (None, compiler)
        self.asan_tests ()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line_malloc = line_number('main.c', '// malloc line')
        self.line_malloc2 = line_number('main.c', '// malloc2 line')
        self.line_free = line_number('main.c', '// free line')
        self.line_breakpoint = line_number('main.c', '// break line')

    def asan_tests (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe, patterns = [ "Current executable set to .*a.out" ])

        self.runCmd("breakpoint set -f main.c -l %d" % self.line_breakpoint)

        # "memory history" command should not work without a process
        self.expect("memory history 0",
            error = True,
            substrs = ["invalid process"])

        self.runCmd("run")

        # ASan will relaunch the process to insert its library.
        self.expect("thread list", "Process should be stopped due to exec.",
            substrs = ['stopped', 'stop reason = '])

        self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

        # test that the ASan dylib is present
        self.expect("image lookup -n __asan_describe_address", "__asan_describe_address should be present",
            substrs = ['1 match found'])

        # test the 'memory history' command
        self.expect("memory history 'pointer'",
            substrs = [
                'Memory allocated at', 'a.out`f1', 'main.c:%d' % self.line_malloc,
                'Memory deallocated at', 'a.out`f2', 'main.c:%d' % self.line_free])

        # do the same using SB API
        process = self.dbg.GetSelectedTarget().process
        val = process.GetSelectedThread().GetSelectedFrame().EvaluateExpression("pointer")
        addr = val.GetValueAsUnsigned()
        threads = process.GetHistoryThreads(addr);
        self.assertEqual(threads.GetSize(), 2)
        
        history_thread = threads.GetThreadAtIndex(0)
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(), "main.c")
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetLine(), self.line_free)
        
        history_thread = threads.GetThreadAtIndex(1)
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(), "main.c")
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetLine(), self.line_malloc)

        # let's free the container (SBThreadCollection) and see if the SBThreads still live
        threads = None
        self.assertTrue(history_thread.num_frames >= 2)
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetFileSpec().GetFilename(), "main.c")
        self.assertEqual(history_thread.frames[1].GetLineEntry().GetLine(), self.line_malloc)

        # now let's break when an ASan report occurs and try the API then
        self.runCmd("breakpoint set -n __asan_report_error")

        self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

        # make sure the 'memory history' command still works even when we're generating a report now
        self.expect("memory history 'another_pointer'",
            substrs = [
                'Memory allocated at', 'a.out`f1', 'main.c:%d' % self.line_malloc2])
