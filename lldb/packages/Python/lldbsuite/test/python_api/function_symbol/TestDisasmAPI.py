"""
Test retrieval of SBAddress from function/symbol, disassembly, and SBAddress APIs.
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class DisasmAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number('main.c', '// Find the line number for breakpoint 1 here.')
        self.line2 = line_number('main.c', '// Find the line number for breakpoint 2 here.')

    @add_test_categories(['pyapi'])
    def test(self):
        """Exercise getting SBAddress objects, disassembly, and SBAddress APIs."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create the two breakpoints inside function 'a'.
        breakpoint1 = target.BreakpointCreateByLocation('main.c', self.line1)
        breakpoint2 = target.BreakpointCreateByLocation('main.c', self.line2)
        #print("breakpoint1:", breakpoint1)
        #print("breakpoint2:", breakpoint2)
        self.assertTrue(breakpoint1 and
                        breakpoint1.GetNumLocations() == 1,
                        VALID_BREAKPOINT)
        self.assertTrue(breakpoint2 and
                        breakpoint2.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line1)

        address1 = lineEntry.GetStartAddress()
        #print("address1:", address1)

        # Now call SBTarget.ResolveSymbolContextForAddress() with address1.
        context1 = target.ResolveSymbolContextForAddress(address1, lldb.eSymbolContextEverything)

        self.assertTrue(context1)
        if self.TraceOn():
            print("context1:", context1)

        # Continue the inferior, the breakpoint 2 should be hit.
        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.line2)

        # Verify that the symbol and the function has the same address range per function 'a'.
        symbol = context1.GetSymbol()
        function = frame0.GetFunction()
        self.assertTrue(symbol and function)

        disasm_output = lldbutil.disassemble(target, symbol)
        if self.TraceOn():
            print("symbol:", symbol)
            print("disassembly=>\n", disasm_output)

        disasm_output = lldbutil.disassemble(target, function)
        if self.TraceOn():
            print("function:", function)
            print("disassembly=>\n", disasm_output)

        sa1 = symbol.GetStartAddress()
        #print("sa1:", sa1)
        #print("sa1.GetFileAddress():", hex(sa1.GetFileAddress()))
        #ea1 = symbol.GetEndAddress()
        #print("ea1:", ea1)
        sa2 = function.GetStartAddress()
        #print("sa2:", sa2)
        #print("sa2.GetFileAddress():", hex(sa2.GetFileAddress()))
        #ea2 = function.GetEndAddress()
        #print("ea2:", ea2)
        self.assertTrue(sa1 and sa2 and sa1 == sa2,
                        "The two starting addresses should be the same")

        from lldbsuite.test.lldbutil import get_description
        desc1 = get_description(sa1)
        desc2 = get_description(sa2)
        self.assertTrue(desc1 and desc2 and desc1 == desc2,
                        "SBAddress.GetDescription() API of sa1 and sa2 should return the same string")
