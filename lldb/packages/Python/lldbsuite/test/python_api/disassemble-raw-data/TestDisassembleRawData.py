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
        arch = self.getArchitecture()
        if re.match("mips*el",arch):
            target = self.dbg.CreateTargetWithFileAndTargetTriple ("", "mipsel")
            raw_bytes = bytearray([0x21,0xf0, 0xa0, 0x03])
        elif re.match("mips",arch):
            target = self.dbg.CreateTargetWithFileAndTargetTriple ("", "mips")
            raw_bytes = bytearray([0x03,0xa0, 0xf0, 0x21])
        else:
            target = self.dbg.CreateTargetWithFileAndTargetTriple ("", "x86_64")
            raw_bytes = bytearray([0x48, 0x89, 0xe5])
        
        self.assertTrue(target, VALID_TARGET)
        insts = target.GetInstructions(lldb.SBAddress(0, target), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print()
            print("Raw bytes:    ", [hex(x) for x in raw_bytes])
            print("Disassembled%s" % str(inst))
        if re.match("mips",arch):
            self.assertTrue (inst.GetMnemonic(target) == "move")
            self.assertTrue (inst.GetOperands(target) == '$' + "fp, " + '$' + "sp")
        else:
            self.assertTrue (inst.GetMnemonic(target) == "movq")
            self.assertTrue (inst.GetOperands(target) == '%' + "rsp, " + '%' + "rbp")
