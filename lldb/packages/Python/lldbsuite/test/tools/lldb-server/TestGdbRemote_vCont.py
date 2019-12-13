
import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemote_vCont(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def vCont_supports_mode(self, mode, inferior_args=None):
        # Setup the stub and set the gdb remote command stream.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=inferior_args)
        self.add_vCont_query_packets()

        # Run the gdb remote command stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Pull out supported modes.
        supported_vCont_modes = self.parse_vCont_query_response(context)
        self.assertIsNotNone(supported_vCont_modes)

        # Verify we support the given mode.
        self.assertTrue(mode in supported_vCont_modes)

    def vCont_supports_c(self):
        self.vCont_supports_mode("c")

    def vCont_supports_C(self):
        self.vCont_supports_mode("C")

    def vCont_supports_s(self):
        self.vCont_supports_mode("s")

    def vCont_supports_S(self):
        self.vCont_supports_mode("S")

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_vCont_supports_c_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.vCont_supports_c()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @llgs_test
    def test_vCont_supports_c_llgs(self):
        self.init_llgs_test()
        self.build()
        self.vCont_supports_c()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_vCont_supports_C_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.vCont_supports_C()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @llgs_test
    def test_vCont_supports_C_llgs(self):
        self.init_llgs_test()
        self.build()
        self.vCont_supports_C()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_vCont_supports_s_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.vCont_supports_s()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @llgs_test
    def test_vCont_supports_s_llgs(self):
        self.init_llgs_test()
        self.build()
        self.vCont_supports_s()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_vCont_supports_S_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.vCont_supports_S()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @llgs_test
    def test_vCont_supports_S_llgs(self):
        self.init_llgs_test()
        self.build()
        self.vCont_supports_S()

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_single_step_only_steps_one_instruction_with_Hc_vCont_s_debugserver(
            self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="vCont;s")

    @skipIfWindows # No pty support to test O* & I* notification packets.
    @llgs_test
    @expectedFailureAndroid(
        bugnumber="llvm.org/pr24739",
        archs=[
            "arm",
            "aarch64"])
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        bugnumber="llvm.org/pr24739")
    @skipIf(triple='^mips')
    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    def test_single_step_only_steps_one_instruction_with_Hc_vCont_s_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=True, step_instruction="vCont;s")

    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    @debugserver_test
    def test_single_step_only_steps_one_instruction_with_vCont_s_thread_debugserver(
            self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=False, step_instruction="vCont;s:{thread}")

    @skipIfWindows # No pty support to test O* & I* notification packets.
    @llgs_test
    @expectedFailureAndroid(
        bugnumber="llvm.org/pr24739",
        archs=[
            "arm",
            "aarch64"])
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        bugnumber="llvm.org/pr24739")
    @skipIf(triple='^mips')
    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://27005337")
    def test_single_step_only_steps_one_instruction_with_vCont_s_thread_llgs(
            self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(
            use_Hc_packet=False, step_instruction="vCont;s:{thread}")
