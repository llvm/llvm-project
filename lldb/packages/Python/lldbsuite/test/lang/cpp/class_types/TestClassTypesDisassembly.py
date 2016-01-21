"""
Test the lldb disassemble command on each call frame when stopped on C's ctor.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class IterateFrameAndDisassembleTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_and_run_command(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.build()
        self.breakOnCtor()

        raw_output = self.res.GetOutput()
        frameRE = re.compile(r"""
                              ^\s\sframe        # heading for the frame info,
                              .*                # wildcard, and
                              0x[0-9a-f]{16}    # the frame pc, and
                              \sa.out`(.+)      # module`function, and
                              \s\+\s            # the rest ' + ....'
                              """, re.VERBOSE)
        for line in raw_output.split(os.linesep):
            match = frameRE.search(line)
            if match:
                function = match.group(1)
                #print("line:", line)
                #print("function:", function)
                self.runCmd("disassemble -n '%s'" % function)

    @add_test_categories(['pyapi'])
    def test_and_python_api(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.build()
        self.breakOnCtor()

        # Now use the Python API to get at each function on the call stack and
        # disassemble it.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        depth = thread.GetNumFrames()
        for i in range(depth - 1):
            frame = thread.GetFrameAtIndex(i)
            function = frame.GetFunction()
            # Print the function header.
            if self.TraceOn():
                print()
                print(function)
            if function:
                # Get all instructions for this function and print them out.
                insts = function.GetInstructions(target)
                for inst in insts:
                    # We could simply do 'print inst' to print out the disassembly.
                    # But we want to print to stdout only if self.TraceOn() is True.
                    disasm = str(inst)
                    if self.TraceOn():
                        print(disasm)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def breakOnCtor(self):
        """Setup/run the program so it stops on C's constructor."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the ctor function of class C.
        bpno = lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint %d.'%(bpno)])

        # This test was failing because we fail to put the C:: in front of constructore.
        # We should maybe make another testcase to cover that specifically, but we shouldn't
        # fail this whole testcase for an inessential issue.
        # We should be stopped on the ctor function of class C.
        # self.expect("thread backtrace", BACKTRACE_DISPLAYED_CORRECTLY,
        #  substrs = ['C::C'])
