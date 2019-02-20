from __future__ import print_function


import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteGPacket(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def run_test_g_packet(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $g#67",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "register_bank"}}],
            True)
        self.connect_to_debug_monitor()
        context = self.expect_gdbremote_sequence()
        register_bank = context.get("register_bank")
        self.assertTrue(register_bank[0] != 'E')

        self.test_sequence.add_log_lines(
            ["read packet: $G" + register_bank + "#00",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "G_reply"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertTrue(context.get("G_reply")[0] != 'E')


    @skipIfOutOfTreeDebugserver
    @debugserver_test
    @skipIfDarwinEmbedded
    def test_g_packet_debugserver(self):
        self.init_debugserver_test()
        self.run_test_g_packet()
