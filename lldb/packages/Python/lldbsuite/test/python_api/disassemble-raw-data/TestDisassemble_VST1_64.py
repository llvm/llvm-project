"""
Use lldb Python API to disassemble raw machine code bytes
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Disassemble_VST1_64(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipTestIfFn(
        lambda: True,
        "llvm.org/pr24575: all tests get ERRORs in dotest.py after this")
    @add_test_categories(['pyapi'])
    @no_debug_info_test
    def test_disassemble_invalid_vst_1_64_raw_data(self):
        """Test disassembling invalid vst1.64 raw bytes with the API."""
        # Create a target from the debugger.
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "thumbv7")
        self.assertTrue(target, VALID_TARGET)

        raw_bytes = bytearray([0xf0, 0xb5, 0x03, 0xaf,
                               0x2d, 0xe9, 0x00, 0x0d,
                               0xad, 0xf1, 0x40, 0x04,
                               0x24, 0xf0, 0x0f, 0x04,
                               0xa5, 0x46])

        insts = target.GetInstructions(lldb.SBAddress(), raw_bytes)

        if self.TraceOn():
            print()
            for i in insts:
                print("Disassembled%s" % str(i))

        # Remove the following return statement when the radar is fixed.
        return

        # rdar://problem/11034702
        # VST1 (multiple single elements) encoding?
        # The disassembler should not crash!
        raw_bytes = bytearray([0x04, 0xf9, 0xed, 0x82])

        insts = target.GetInstructions(lldb.SBAddress(), raw_bytes)

        inst = insts.GetInstructionAtIndex(0)

        if self.TraceOn():
            print()
            print("Raw bytes:    ", [hex(x) for x in raw_bytes])
            print("Disassembled%s" % str(inst))

        self.assertTrue(inst.GetMnemonic(target) == "vst1.64")
