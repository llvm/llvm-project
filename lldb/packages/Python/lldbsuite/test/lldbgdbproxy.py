import logging
import os
import os.path
import random

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.gdbclientutils import *
import lldbgdbserverutils
from lldbsuite.support import seven


class GDBProxyTestBase(TestBase):
    """
    Base class for gdbserver proxy tests.

    This class will setup and start a mock GDB server for the test to use.
    It pases through requests to a regular lldb-server/debugserver and
    forwards replies back to the LLDB under test.
    """

    """The gdbserver that we implement."""
    server = None
    """The inner lldb-server/debugserver process that we proxy requests into."""
    monitor_server = None
    monitor_sock = None

    server_socket_class = TCPServerSocket

    DEFAULT_TIMEOUT = 20 * (10 if ("ASAN_OPTIONS" in os.environ) else 1)

    _verbose_log_handler = None
    _log_formatter = logging.Formatter(fmt="%(asctime)-15s %(levelname)-8s %(message)s")

    def setUpBaseLogging(self):
        self.logger = logging.getLogger(__name__)

        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        # log all warnings to stderr
        self._stderr_log_handler = logging.StreamHandler()
        self._stderr_log_handler.setLevel(
            logging.DEBUG if self.TraceOn() else logging.WARNING
        )
        self._stderr_log_handler.setFormatter(self._log_formatter)
        self.logger.addHandler(self._stderr_log_handler)

    def setUp(self):
        TestBase.setUp(self)

        self.setUpBaseLogging()

        if self.isVerboseLoggingRequested():
            # If requested, full logs go to a log file
            log_file_name = self.getLogBasenameForCurrentTest() + "-proxy.log"
            self._verbose_log_handler = logging.FileHandler(log_file_name)
            self._verbose_log_handler.setFormatter(self._log_formatter)
            self._verbose_log_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self._verbose_log_handler)

        if lldbplatformutil.getPlatform() == "macosx":
            self.debug_monitor_exe = lldbgdbserverutils.get_debugserver_exe()
            self.debug_monitor_extra_args = []
        else:
            self.debug_monitor_exe = lldbgdbserverutils.get_lldb_server_exe()
            self.debug_monitor_extra_args = ["gdbserver"]
        self.assertIsNotNone(self.debug_monitor_exe)

        self.server = MockGDBServer(self.server_socket_class())
        self.server.responder = self

    def tearDown(self):
        # TestBase.tearDown will kill the process, but we need to kill it early
        # so its client connection closes and we can stop the server before
        # finally calling the base tearDown.
        if self.process() is not None:
            self.process().Kill()
        self.server.stop()

        self.logger.removeHandler(self._verbose_log_handler)
        self._verbose_log_handler = None
        self.logger.removeHandler(self._stderr_log_handler)
        self._stderr_log_handler = None

        TestBase.tearDown(self)

    def isVerboseLoggingRequested(self):
        # We will report our detailed logs if the user requested that the "gdb-remote" channel is
        # logged.
        return any(("gdb-remote" in channel) for channel in lldbtest_config.channels)

    def connect(self, target):
        """
        Create a process by connecting to the mock GDB server.
        """
        self.prep_debug_monitor_and_inferior()
        self.server.start()

        listener = self.dbg.GetListener()
        error = lldb.SBError()
        process = target.ConnectRemote(
            listener, self.server.get_connect_url(), "gdb-remote", error
        )
        self.assertTrue(error.Success(), error.description)
        self.assertTrue(process, PROCESS_IS_VALID)
        return process

    def prep_debug_monitor_and_inferior(self):
        inferior_exe_path = self.getBuildArtifact("a.out")
        self.connect_to_debug_monitor([inferior_exe_path])
        self.assertIsNotNone(self.monitor_server)
        self.initial_handshake()

    def initial_handshake(self):
        self.monitor_server.send_packet(seven.bitcast_to_bytes("+"))
        reply = seven.bitcast_to_string(self.monitor_server.get_normal_packet())
        self.assertEqual(reply, "+")
        self.monitor_server.send_packet(seven.bitcast_to_bytes("QStartNoAckMode"))
        reply = seven.bitcast_to_string(self.monitor_server.get_normal_packet())
        self.assertEqual(reply, "+")
        reply = seven.bitcast_to_string(self.monitor_server.get_normal_packet())
        self.assertEqual(reply, "OK")
        self.monitor_server.set_validate_checksums(False)
        self.monitor_server.send_packet(seven.bitcast_to_bytes("+"))
        reply = seven.bitcast_to_string(self.monitor_server.get_normal_packet())
        self.assertEqual(reply, "+")

    def get_debug_monitor_command_line_args(self, connect_address, launch_args):
        return (
            self.debug_monitor_extra_args
            + ["--reverse-connect", connect_address]
            + launch_args
        )

    def launch_debug_monitor(self, launch_args):
        family, type, proto, _, addr = socket.getaddrinfo(
            "localhost", 0, proto=socket.IPPROTO_TCP
        )[0]
        sock = socket.socket(family, type, proto)
        sock.settimeout(self.DEFAULT_TIMEOUT)
        sock.bind(addr)
        sock.listen(1)
        addr = sock.getsockname()
        connect_address = "[{}]:{}".format(*addr)

        commandline_args = self.get_debug_monitor_command_line_args(
            connect_address, launch_args
        )

        # Start the server.
        self.logger.info(f"Spawning monitor {commandline_args}")
        monitor_process = self.spawnSubprocess(
            self.debug_monitor_exe, commandline_args, install_remote=False
        )
        self.assertIsNotNone(monitor_process)

        self.monitor_sock = sock.accept()[0]
        self.monitor_sock.settimeout(self.DEFAULT_TIMEOUT)
        return monitor_process

    def connect_to_debug_monitor(self, launch_args):
        monitor_process = self.launch_debug_monitor(launch_args)
        # Turn off checksum validation because debugserver does not produce
        # correct checksums.
        self.monitor_server = lldbgdbserverutils.Server(
            self.monitor_sock, monitor_process
        )

    def respond(self, packet):
        """Subclasses can override this to change how packets are handled."""
        return self.pass_through(packet)

    def pass_through(self, packet):
        self.logger.info(f"Sending packet {packet}")
        self.monitor_server.send_packet(seven.bitcast_to_bytes(packet))
        reply = seven.bitcast_to_string(self.monitor_server.get_normal_packet())
        self.logger.info(f"Received reply {reply}")
        return reply
