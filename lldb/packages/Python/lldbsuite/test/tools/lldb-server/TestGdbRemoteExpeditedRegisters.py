from __future__ import print_function



import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteExpeditedRegisters(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def gather_expedited_registers(self):
        # Setup the stub and set the gdb remote command stream.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:2"])
        self.test_sequence.add_log_lines([
            # Start up the inferior.
            "read packet: $c#63",
            # Immediately tell it to stop.  We want to see what it reports.
            "read packet: {}".format(chr(3)),
            {"direction":"send", "regex":r"^\$T([0-9a-fA-F]+)([^#]+)#[0-9a-fA-F]{2}$", "capture":{1:"stop_result", 2:"key_vals_text"} },
            ], True)

        # Run the gdb remote command stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Pull out expedited registers.
        key_vals_text = context.get("key_vals_text")
        self.assertIsNotNone(key_vals_text)

        expedited_registers = self.extract_registers_from_stop_notification(key_vals_text)
        self.assertIsNotNone(expedited_registers)

        return expedited_registers

    def stop_notification_contains_generic_register(self, generic_register_name):
        # Generate a stop reply, parse out expedited registers from stop notification.
        expedited_registers = self.gather_expedited_registers()
        self.assertIsNotNone(expedited_registers)
        self.assertTrue(len(expedited_registers) > 0)

        # Gather target register infos.
        reg_infos = self.gather_register_infos()

        # Find the generic register.
        reg_info = self.find_generic_register_with_name(reg_infos, generic_register_name)
        self.assertIsNotNone(reg_info)

        # Ensure the expedited registers contained it.
        self.assertTrue(reg_info["lldb_register_index"] in expedited_registers)
        # print("{} reg_info:{}".format(generic_register_name, reg_info))

    def stop_notification_contains_any_registers(self):
        # Generate a stop reply, parse out expedited registers from stop notification.
        expedited_registers = self.gather_expedited_registers()
        # Verify we have at least one expedited register.
        self.assertTrue(len(expedited_registers) > 0)

    @debugserver_test
    def test_stop_notification_contains_any_registers_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_any_registers()

    @llgs_test
    def test_stop_notification_contains_any_registers_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_any_registers()

    def stop_notification_contains_no_duplicate_registers(self):
        # Generate a stop reply, parse out expedited registers from stop notification.
        expedited_registers = self.gather_expedited_registers()
        # Verify no expedited register was specified multiple times.
        for (reg_num, value) in list(expedited_registers.items()):
            if (type(value) == list) and (len(value) > 0):
                self.fail("expedited register number {} specified more than once ({} times)".format(reg_num, len(value)))

    @debugserver_test
    def test_stop_notification_contains_no_duplicate_registers_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_no_duplicate_registers()

    @llgs_test
    def test_stop_notification_contains_no_duplicate_registers_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_no_duplicate_registers()

    def stop_notification_contains_pc_register(self):
        self.stop_notification_contains_generic_register("pc")

    @debugserver_test
    def test_stop_notification_contains_pc_register_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_pc_register()

    @llgs_test
    def test_stop_notification_contains_pc_register_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_pc_register()

    def stop_notification_contains_fp_register(self):
        self.stop_notification_contains_generic_register("fp")

    @debugserver_test
    def test_stop_notification_contains_fp_register_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_fp_register()

    @llgs_test
    def test_stop_notification_contains_fp_register_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_fp_register()

    def stop_notification_contains_sp_register(self):
        self.stop_notification_contains_generic_register("sp")

    @debugserver_test
    def test_stop_notification_contains_sp_register_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_sp_register()

    @llgs_test
    def test_stop_notification_contains_sp_register_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.stop_notification_contains_sp_register()
