"""
Test the solution to issue 11581.
valobj.AddressOf() returns None when an address is
expected in a SyntheticChildrenProvider
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Issue11581TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_11581_commands(self):
        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type synthetic clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """valobj.AddressOf() should return correct values."""
        self.build()

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here.', lldb.SBFileSpec("main.cpp", False))

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, "Created a process.")
        self.assertTrue(
            process.GetState() == lldb.eStateStopped,
            "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)
        self.assertTrue(len(thread_list) == 1)
        thread = thread_list[0]

        self.runCmd("command script import --allow-reload s11588.py")
        self.runCmd(
            "type synthetic add --python-class s11588.Issue11581SyntheticProvider StgClosure")

        self.expect("expr --show-types -- *((StgClosure*)(r14-1))",
                    substrs=["(StgClosure) $",
                             "(StgClosure *) &$", "0x",
                             "addr = ",
                             "load_address = "])

        # register r14 is an x86_64 extension let's skip this part of the test
        # if we are on a different architecture
        if self.getArchitecture() == 'x86_64':
            target = lldb.debugger.GetSelectedTarget()
            process = target.GetProcess()
            frame = process.GetSelectedThread().GetSelectedFrame()
            pointer = frame.FindVariable("r14")
            addr = pointer.GetValueAsUnsigned(0)
            self.assertTrue(addr != 0, "could not read pointer to StgClosure")
            addr = addr - 1
            self.runCmd("register write r14 %d" % addr)
            self.expect(
                "register read r14", substrs=[
                    "0x", hex(addr)[
                        2:].rstrip("L")])  # Remove trailing 'L' if it exists
            self.expect("expr --show-types -- *(StgClosure*)$r14",
                        substrs=["(StgClosure) $",
                                 "(StgClosure *) &$", "0x",
                                 "addr = ",
                                 "load_address = ",
                                 hex(addr)[2:].rstrip("L"),
                                 str(addr)])
