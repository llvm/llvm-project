"""
Use lldb Python API to disassemble raw machine code bytes
"""

from __future__ import print_function



import os, time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class DisassembleRawDataTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_disassemble_raw_data(self):
        """Test disassembling raw bytes with the API."""
        # Create a target from the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple ("", "x86_64")
        self.assertTrue(target, VALID_TARGET)

        raw_bytes = bytearray([0x48, 0x89, 0xe5])

        insts = target.GetInstructions(lldb.SBAddress(0, target), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print()
            print("Raw bytes:    ", [hex(x) for x in raw_bytes])
            print("Disassembled%s" % str(inst))
 
        self.assertTrue (inst.GetMnemonic(target) == "movq")
        self.assertTrue (inst.GetOperands(target) == '%' + "rsp, " + '%' + "rbp")
