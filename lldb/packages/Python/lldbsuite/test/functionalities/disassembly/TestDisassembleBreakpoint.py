"""
Test some lldb command abbreviations.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DisassemblyTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="function names print fully demangled instead of name-only")
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe,
                    patterns=["Current executable set to .*a.out.*"])

        self.runCmd("dis -n main")
        disassembly_before_break = self.res.GetOutput().splitlines()

        match_object = lldbutil.run_break_set_command(self, "br s -n sum")
        lldbutil.check_breakpoint_result(
            self,
            match_object,
            symbol_name='sum',
            symbol_match_exact=False,
            num_locations=1)

        self.expect("run",
                    patterns=["Process .* launched: "])

        self.runCmd("dis -n main")
        disassembly_after_break = self.res.GetOutput().splitlines()

        # make sure all assembly instructions are the same as the original
        # instructions before inserting breakpoints.
        self.assertEqual(len(disassembly_before_break),
                         len(disassembly_after_break))

        for dis_inst_before, dis_inst_after in \
                zip(disassembly_before_break, disassembly_after_break):
            inst_before = dis_inst_before.split(':')[-1]
            inst_after = dis_inst_after.split(':')[-1]
            self.assertEqual(inst_before, inst_after)
