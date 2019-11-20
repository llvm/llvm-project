from __future__ import print_function



import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteProcessInfo(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def qProcessInfo_returns_running_process(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id looks reasonable.
        pid_text = process_info.get("pid")
        self.assertIsNotNone(pid_text)
        pid = int(pid_text, base=16)
        self.assertNotEqual(0, pid)

        # If possible, verify that the process is running.
        self.assertTrue(lldbgdbserverutils.process_is_running(pid, True))

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_qProcessInfo_returns_running_process_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.qProcessInfo_returns_running_process()

    @llgs_test
    def test_qProcessInfo_returns_running_process_llgs(self):
        self.init_llgs_test()
        self.build()
        self.qProcessInfo_returns_running_process()

    def attach_commandline_qProcessInfo_reports_correct_pid(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.assertIsNotNone(procs)
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence(timeout_seconds=self._DEFAULT_TIMEOUT)
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id matches what we expected.
        pid_text = process_info.get('pid', None)
        self.assertIsNotNone(pid_text)
        reported_pid = int(pid_text, base=16)
        self.assertEqual(reported_pid, procs["inferior"].pid)

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_attach_commandline_qProcessInfo_reports_correct_pid_debugserver(
            self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_attach()
        self.attach_commandline_qProcessInfo_reports_correct_pid()

    @expectedFailureNetBSD
    @llgs_test
    def test_attach_commandline_qProcessInfo_reports_correct_pid_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_attach()
        self.attach_commandline_qProcessInfo_reports_correct_pid()

    def qProcessInfo_reports_valid_endian(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the process id looks reasonable.
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)
        self.assertTrue(endian in ["little", "big", "pdp"])

    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_qProcessInfo_reports_valid_endian_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.qProcessInfo_reports_valid_endian()

    @llgs_test
    def test_qProcessInfo_reports_valid_endian_llgs(self):
        self.init_llgs_test()
        self.build()
        self.qProcessInfo_reports_valid_endian()

    def qProcessInfo_contains_keys(self, expected_key_set):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the expected keys are present and non-None within the process
        # info.
        missing_key_set = set()
        for expected_key in expected_key_set:
            if expected_key not in process_info:
                missing_key_set.add(expected_key)

        self.assertEqual(
            missing_key_set,
            set(),
            "the listed keys are missing in the qProcessInfo result")

    def qProcessInfo_does_not_contain_keys(self, absent_key_set):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_process_info_collection_packets()

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Ensure the unexpected keys are not present
        unexpected_key_set = set()
        for unexpected_key in absent_key_set:
            if unexpected_key in process_info:
                unexpected_key_set.add(unexpected_key)

        self.assertEqual(
            unexpected_key_set,
            set(),
            "the listed keys were present but unexpected in qProcessInfo result")

    @skipUnlessDarwin
    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_qProcessInfo_contains_cputype_cpusubtype_debugserver_darwin(self):
        self.init_debugserver_test()
        self.build()
        self.qProcessInfo_contains_keys(set(['cputype', 'cpusubtype']))

    @skipUnlessDarwin
    @llgs_test
    def test_qProcessInfo_contains_cputype_cpusubtype_llgs_darwin(self):
        self.init_llgs_test()
        self.build()
        self.qProcessInfo_contains_keys(set(['cputype', 'cpusubtype']))

    @llgs_test
    def test_qProcessInfo_contains_triple_ppid_llgs(self):
        self.init_llgs_test()
        self.build()
        self.qProcessInfo_contains_keys(set(['triple', 'parent-pid']))

    @skipUnlessDarwin
    @debugserver_test
    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def test_qProcessInfo_does_not_contain_triple_debugserver_darwin(self):
        self.init_debugserver_test()
        self.build()
        # We don't expect to see triple on darwin.  If we do, we'll prefer triple
        # to cputype/cpusubtype and skip some darwin-based ProcessGDBRemote ArchSpec setup
        # for the remote Host and Process.
        self.qProcessInfo_does_not_contain_keys(set(['triple']))

    @skipUnlessDarwin
    @llgs_test
    def test_qProcessInfo_does_not_contain_triple_llgs_darwin(self):
        self.init_llgs_test()
        self.build()
        # We don't expect to see triple on darwin.  If we do, we'll prefer triple
        # to cputype/cpusubtype and skip some darwin-based ProcessGDBRemote ArchSpec setup
        # for the remote Host and Process.
        self.qProcessInfo_does_not_contain_keys(set(['triple']))

    @skipIfDarwin
    @llgs_test
    def test_qProcessInfo_does_not_contain_cputype_cpusubtype_llgs(self):
        self.init_llgs_test()
        self.build()
        self.qProcessInfo_does_not_contain_keys(set(['cputype', 'cpusubtype']))
