from __future__ import print_function


import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteSingleStep(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @debugserver_test
    def test_single_step_only_steps_one_instruction_with_s_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="s")

    @llgs_test
    @expectedFailureAndroid(
        bugnumber="llvm.org/pr24739",
        archs=[
            "arm",
            "aarch64"])
    @expectedFailureAll(
        oslist=["linux"],
        archs=[
            "arm",
            "aarch64"],
        bugnumber="llvm.org/pr24739")
    @skipIf(triple='^mips')
    def test_single_step_only_steps_one_instruction_with_s_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="s")
