"""
Tests the binary ($x) and hex ($m) memory read packets of the remote stub
"""

from __future__ import print_function


import binascii
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbplatformutil


class MemoryReadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessPlatform(lldbplatformutil.getDarwinOSTriples() + ["linux"])
    def test_memory_read(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        lldbutil.run_break_set_by_symbol(self, "main")

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(
            process.GetState(),
            lldb.eStateStopped,
            "Process is stopped")

        pc = process.GetSelectedThread().GetSelectedFrame().GetPC()
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            error = lldb.SBError()
            memory = process.ReadMemory(pc, size, error)
            self.assertTrue(error.Success())
            # Results in trying to write non-printable characters to the session log.
            # self.match("process plugin packet send x%x,%x" % (pc, size), ["response:", memory])
            self.match("process plugin packet send m%x,%x" %
                       (pc, size), ["response:", binascii.hexlify(memory)])

        process.Continue()
        self.assertEqual(
            process.GetState(),
            lldb.eStateExited,
            "Process exited")
