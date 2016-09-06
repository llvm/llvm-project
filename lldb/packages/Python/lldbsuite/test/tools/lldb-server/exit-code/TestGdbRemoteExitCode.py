from __future__ import print_function

# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase


class TestGdbRemoteExitCode(GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    FAILED_LAUNCH_CODE = "E08"

    def get_launch_fail_reason(self):
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            ["read packet: $qLaunchSuccess#00"],
            True)
        self.test_sequence.add_log_lines(
            [{"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "launch_result"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        return context.get("launch_result")[1:]

    def start_inferior(self):
        launch_args = self.install_and_create_launch_args()

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        self.add_no_ack_remote_stream()
        self.test_sequence.add_log_lines(
            ["read packet: %s" % lldbgdbserverutils.build_gdbremote_A_packet(
                launch_args)],
            True)
        self.test_sequence.add_log_lines(
            [{"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "A_result"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        launch_result = context.get("A_result")
        self.assertIsNotNone(launch_result)
        if launch_result == self.FAILED_LAUNCH_CODE:
            fail_reason = self.get_launch_fail_reason()
            self.fail("failed to launch inferior: " + fail_reason)

    @debugserver_test
    def test_start_inferior_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.start_inferior()

    @llgs_test
    def test_start_inferior_llgs(self):
        self.init_llgs_test()
        self.build()
        self.start_inferior()

    def inferior_exit_0(self):
        launch_args = self.install_and_create_launch_args()

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8",
             "send packet: $W00#00"],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    def test_inferior_exit_0_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.inferior_exit_0()

    @llgs_test
    def test_inferior_exit_0_llgs(self):
        self.init_llgs_test()
        self.build()
        self.inferior_exit_0()

    def inferior_exit_42(self):
        launch_args = self.install_and_create_launch_args()

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        RETVAL = 42

        # build launch args
        launch_args += ["retval:%d" % RETVAL]

        self.add_no_ack_remote_stream()
        self.add_verified_launch_packets(launch_args)
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8",
             "send packet: $W{0:02x}#00".format(RETVAL)],
            True)

        self.expect_gdbremote_sequence()

    @debugserver_test
    def test_inferior_exit_42_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.inferior_exit_42()

    @llgs_test
    def test_inferior_exit_42_llgs(self):
        self.init_llgs_test()
        self.build()
        self.inferior_exit_42()
