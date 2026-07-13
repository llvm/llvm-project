import logging
import os
from pathlib import Path
from typing import Any, Final, Optional, TypeVar, Union, cast

from lldbsuite.test.lldbtest import Base, LLDBTestCaseFactory, is_exe

from .dap_types import AnyResponse, ErrorResponse, Response
from .session_helpers import DAPTestSession
from .utils import DebugAdapter, DebugAdapterOptions


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to a bool following LLVM's CLI argument parsing."""

    val = val.lower()
    if val in {"false", "0", "no", "off"}:
        return False
    return True


T = TypeVar("T")


class DAPTestCaseBase(Base, metaclass=LLDBTestCaseFactory):
    """Base test case for DAP tests"""

    NO_DEBUG_INFO_TESTCASE = True
    DEFAULT_TIMEOUT: Final[float] = 500.0 if "ASAN_OPTIONS" in os.environ else 50.0

    USE_DEFAULT_DEBUG_ADAPTER: bool = True
    """Subclasses can set this to true to avoid creating a debug adapter is will not be used."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.run_as_server: bool = strtobool(os.getenv("LLDBDAP_RUN_AS_SERVER", "false"))

    def setUp(self):
        super().setUp()
        self.setUpBaseLogging()

        self._debug_adapter_count: int = 0
        if self.USE_DEFAULT_DEBUG_ADAPTER:
            self.__create_default_debug_adapter()

    def setUpBaseLogging(self):
        self.logger = logging.getLogger(f"lldb_dap.{__name__}")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        log_path = f"{self.getLogBasenameForCurrentTest()}-test_dap.log"
        handler = logging.FileHandler(log_path, mode="w")

        # The Log name gets quite long and becomes noise. use the last log scope.
        class _ShortNameFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                record.short_name = record.name.rsplit(".", 1)[-1]
                return super().format(record)

        handler.setFormatter(
            _ShortNameFormatter(
                "%(asctime)s.%(msecs)03d %(levelname)-5s (%(short_name)s) %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

        def close_log():
            self.logger.removeHandler(handler)
            handler.close()

        self.addTearDownHook(close_log)

    def __create_default_debug_adapter(self):
        self.assertFalse(hasattr(self, "adapter"), "A default adapter already exists.")

        if self.run_as_server:
            self.adapter = self.create_server_debug_adapter(
                DebugAdapterOptions(cwd=self.getBuildDir()),
                connection="listen://localhost:0",
                connection_timeout=10,
            )
        else:
            self.adapter = self.create_stdio_debug_adapter(
                DebugAdapterOptions(cwd=self.getBuildDir())
            )

    def create_session(
        self,
        adapter: Optional[DebugAdapter] = None,
        disconnect_automatically: bool = True,
    ) -> DAPTestSession:
        if adapter is None:
            self.assertIsNotNone(self.adapter, "expected we already have an adapter.")
            adapter = self.adapter
        self.assertTrue(adapter.is_alive, "expected adapter process is alive.")

        build_dir = Path(self.getBuildDir())
        session = DAPTestSession(
            self,
            build_dir,
            adapter,
            message_timeout=self.DEFAULT_TIMEOUT,
            process_spawner=self.spawnSubprocess,  # type: ignore
            logger=self.logger,
        )

        def cleanup_session():
            if disconnect_automatically:
                self.logger.debug("Automatically disconnecting.")
                session.disconnect(terminateDebuggee=True)
            session.stop()

        session.start()
        self.addTearDownHook(cleanup_session)
        return session

    def build_and_create_session(
        self,
        adapter: Optional[DebugAdapter] = None,
        disconnect_automatically: bool = True,
    ) -> DAPTestSession:
        self.build()
        return self.create_session(adapter, disconnect_automatically)

    def create_debug_adapter(
        self, adapter_options: DebugAdapterOptions
    ) -> DebugAdapter:
        self.assertTrue(
            is_exe(self.lldbDAPExec),
            f"lldb-dap must exist and be executable. path: {self.lldbDAPExec}",
        )

        if adapter_options.log_file:
            log_file = adapter_options.log_file
        else:
            count = self._debug_adapter_count
            suffix = f"-{count}" if count else ""
            log_file = f"{self.getLogBasenameForCurrentTest()}-dap{suffix}.log"

        self._debug_adapter_count += 1
        cwd = adapter_options.cwd or self.getBuildDir()
        pre_init_commands = self.setUpCommands()

        adapter_options = adapter_options.clone(
            log_file=log_file, cwd=cwd, pre_init_commands=pre_init_commands
        )
        lldb_dap_exec = self.expect_not_none(self.lldbDAPExec)
        adapter = DebugAdapter(executable=lldb_dap_exec, opts=adapter_options)
        self.assertTrue(adapter.is_alive, "adapter should be running after creation.")

        def cleanup_adapter():
            if adapter.is_alive:
                adapter.kill()

        self.addTearDownHook(cleanup_adapter)
        return adapter

    def create_stdio_debug_adapter(
        self, adapter_options: Optional[DebugAdapterOptions] = None
    ) -> DebugAdapter:
        """Forces the adapter to stdio mode. the DebugAdapter class handles the validation"""
        adapter_options = adapter_options or DebugAdapterOptions()
        self.assertIsNone(
            adapter_options.connection, "'connection' cannot be used with stdio mode."
        )

        adapter = self.create_debug_adapter(adapter_options)
        self.assertFalse(adapter.is_server, "adapter should be using stdio.")
        return adapter

    def create_server_debug_adapter(
        self,
        adapter_options: Optional[DebugAdapterOptions] = None,
        *,
        connection: str,
        connection_timeout: int,
    ) -> DebugAdapter:
        """Forces the adapter to server mode. the DebugAdapter class handles the validation."""
        adapter_options = adapter_options or DebugAdapterOptions()
        adapter_options = adapter_options.clone(
            connection=connection,
            connection_timeout=connection_timeout,
        )
        adapter = self.create_debug_adapter(adapter_options)
        self.assertTrue(adapter.is_server, "adapter should run as a server.")
        return adapter

    def expect_not_none(self, value: Optional[T], msg: Any = None) -> T:
        """Convenience function to narrow fields that are optional, as most DAP types are."""
        self.assertIsNotNone(value, msg=msg)
        return cast(T, value)

    def expect_error(
        self, value: Union[Response, ErrorResponse], msg: Any = None
    ) -> ErrorResponse:
        """Convenience function for narrowing a response Union to `ErrorResponse`."""
        self.assertIsInstance(value, ErrorResponse, msg=msg)
        self.assertFalse(value.success)
        return cast(ErrorResponse, value)

    def expect_success(
        self, value: Union[AnyResponse, ErrorResponse], msg: Any = None
    ) -> AnyResponse:
        """Convenience function for narrowing a response Union to the success type."""
        self.assertNotIsInstance(value, ErrorResponse, msg=msg)
        self.assertTrue(value.success)
        return cast(AnyResponse, value)
