# FIXME: remove when LLDB_MINIMUM_PYTHON_VERSION > 3.8
from __future__ import annotations

import bisect
import contextlib
import io
import itertools
import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, replace
from pprint import pformat
from typing import IO, Callable, Optional, Protocol, Tuple, Type, runtime_checkable

from .dap_types import (
    AnyEvent,
    DAPError,
    Event,
    MessageType,
    RawMessage,
    Request,
    Response,
)


# See lldbtest.Base.spawnSubprocess, which should help ensure any processes
# created by the DAP client are terminated correctly when the test ends.
class SubProcessSpawner(Protocol):
    def __call__(
        self,
        executable: str,
        args: list[str] | None = None,
        extra_env: list[str] | None = None,
        install_remote: bool = True,
        **kwargs,
    ) -> subprocess.Popen[bytes]:
        ...


@dataclass(frozen=True)
class DebugAdapterOptions:
    """The options passed when spawning the debug adapter."""

    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    pre_init_commands: Optional[list[str]] = None
    log_file: Optional[str] = None
    # sever_mode related options.
    connection: Optional[str] = None
    connection_timeout: Optional[int] = None

    @property
    def run_as_server(self):
        return self.connection is not None

    def clone(self, **kwargs) -> DebugAdapterOptions:
        """Returns a copy with the given fields overridden."""
        return replace(self, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}: {pformat(asdict(self), indent=2, compact=True)}"

    def __post_init__(self):
        # Check connection options is not in args.
        if "--connection" in self.args or "--connection-timeout" in self.args:
            raise DAPError(
                f"--connection in adapter options, use the connection field instead {self}"
            )

        if not self.run_as_server and self.connection_timeout is not None:
            raise DAPError(
                f"'--connection-timeout' option can only be used when a connection is specified: {self}"
            )


class DebugAdapter:
    """Spawns and owns the lifetime of lldb-dap binary"""

    _listening_uri: Optional[str]

    def __init__(self, executable: str, opts: DebugAdapterOptions):
        self.executable = executable
        self._connection_count = 0
        self._is_server = opts.run_as_server

        # Setup the process args.
        process_args = [self.executable]
        process_args.extend(opts.args)

        if pre_init_commands := opts.pre_init_commands:
            for command in pre_init_commands:
                process_args.extend(["--pre-init-command", command])

        # Verify we are using the correct args in stdio or server mode.
        if opts.run_as_server:
            process_args.extend(["--connection", opts.connection])  # type: ignore
            if opts.connection_timeout:
                connection_timeout = str(opts.connection_timeout)
                process_args.extend(["--connection-timeout", connection_timeout])

        # Setup process environment.
        process_env = os.environ.copy()
        process_env.update(opts.env)
        if log_file := opts.log_file:
            process_env["LLDBDAP_LOG"] = log_file

        self._process = subprocess.Popen(
            process_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            cwd=opts.cwd,
        )
        assert self.is_alive, "expected running process"

        if self.is_server:
            self._listening_uri = self._read_listening_uri()
        else:
            self._listening_uri = None

    def create_connection(self) -> DAPConnection:
        if self.is_server:
            assert self._listening_uri is not None
            transport = _SocketTransport(uri=self._listening_uri)
        else:
            if self._connection_count > 0:
                raise DAPError("Cannot create multiple connections in stdio mode")
            transport = _StdioTransport(self._process)

        count = self._connection_count
        connection_id = f"conn{count}" if self.is_server else "stdio"
        self._connection_count += 1
        return DAPConnection(connection_id, transport)

    @property
    def is_server(self):
        return self._is_server

    @property
    def is_alive(self):
        return self._process.poll() is None

    @property
    def process(self):
        return self._process

    def kill(self):
        self._process.terminate()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()

    def _read_listening_uri(self) -> str:
        # lldb-dap will print the listening address once the listener is
        # made to stdout. The listener is formatted like
        # `connection://host:port` or `unix-connection:///path`.
        expected_prefix = "Listening for: "
        process_stdout = self._process.stdout
        if process_stdout is None:
            raise AttributeError("expected the process stdout to be a PIPE")

        out = process_stdout.readline().decode()
        if not out:
            # Check if there is a message in stderr.
            err = ""
            with contextlib.suppress(Exception):
                if process_stderr := self.process.stderr:
                    err = process_stderr.read().decode()
            raise EOFError(
                f"Unexpected End of file for process {self.process.args},\n"
                f"process stderr: {err}"
            )

        if not out.startswith(expected_prefix):
            raise ValueError(
                "lldb-dap failed to print listening address, "
                f"expected '{expected_prefix}', got '{out}'"
            )

        # FIXME: use `str.removeprefix` when LLDB_MINIMUM_PYTHON_VERSION > 3.8
        out = out[len(expected_prefix) :]

        # If the listener expanded into multiple addresses, use the first.
        uri = out.rstrip("\r\n").split(",", 1)[0]
        return uri


class EventHistory:
    """Thread-safe event log that tests block against to observe the adapter.

    Every event the debug adapter sends is recorded here by the read
    thread, in the order it arrived. Tests don't read the log directly,
    they call one of the `wait_for_*` methods, which block until a matching
    event has been recorded.

    Args:
        timeout: Default timeout in seconds for `wait_for_*` functions.

    Example:
    Wait for a stop after stepping, without racing the adapter.

    >>> step_resp = session.step_in(thread_id=1)
    >>> # History will only check for events after the step_response sequence.
    >>> stopped = history.wait_for_event(StoppedEvent, after=step_resp)

    Wait for any of several events (either is an acceptable outcome).
    >>> end = history.wait_for_any_event((StoppedEvent, TerminatedEvent), after=continue_resp)

    Narrow with a predicate.

    >>> hit = history.wait_for_event(
    ...    StoppedEvent,
    ...    after=launch_resp,
    ...    until=lambda e: e.body.reason == "breakpoint",
    ... )

    Find the first Initialized event from the start of the history.
    >>> init_event = history.wait_for_earliest_event(InitializedEvent)
    """

    def __init__(self, timeout: float):
        self._sequences: list[int] = []
        self._events: list[Event] = []
        self._new_event_condition = threading.Condition()
        self._timeout: float = timeout

        self._is_closed: bool = False
        self._closed_reason: Optional[Exception] = None

    @property
    def is_closed(self):
        with self._new_event_condition:
            return self._is_closed

    def close(self, reason: Optional[Exception] = None):
        """Close the history and wake all pending waiters.

        After closing, `record` raises `DAPError` and any in-flight
        `wait_for_*` call raises `DAPError` instead of timing out. This
        is called when the adapter disconnects or the session ends so
        tests do not block for the full default timeout.

        Args:
            reason: Optional exception describing why the history was
                closed. When set, it is included in the error raised by
                waiters so they can see the underlying cause.
        """
        with self._new_event_condition:
            if self._is_closed:
                raise DAPError(
                    f"history already closed with exception {self._closed_reason}"
                    f"trying to close again with {reason}."
                )
            self._is_closed = True
            self._closed_reason = reason
            self._new_event_condition.notify_all()

    def record(self, new_event: Event):
        """Record an event in the history.

        Enforces recording event in sequential order.
        Raises:
            DAPError: If the history has been closed or If the new event
                seq is not greater than the last recorded event's `seq`.
        """
        new_seq: int = new_event.seq
        with self._new_event_condition:
            if self._is_closed:
                raise DAPError(
                    "Cannot record in EventHistory: session is closed."
                ) from self._closed_reason

            if len(self._sequences) > 0:
                # History must be sequential.
                last_seen_seq = self._sequences[-1]
                if new_seq <= last_seen_seq:
                    raise DAPError(
                        f"event: '{new_event.event}' seq '{new_seq}' is older than last event: "
                        f"'{self._events[-1].event}' seq: '{last_seen_seq}'"
                    )

            self._sequences.append(new_seq)
            self._events.append(new_event)

            # Sanity check.
            assert len(self._sequences) == len(self._events)
            self._new_event_condition.notify_all()

    def wait_for_earliest_event(
        self,
        event_type: Type[AnyEvent],
        *,
        until: Optional[Callable[[AnyEvent], bool]] = None,
        timeout: Optional[float] = None,
        timeout_msg: Optional[str] = None,
    ) -> AnyEvent:
        """Wait for the earliest event of `event_type` in the history.

        Searches from the beginning of the log (`seq` 0), so already-received
        events count. Use this when a test wants the first event of a given
        kind regardless of when it arrived.

        Raises the same exceptions as `wait_for_event`.
        """
        assert issubclass(event_type, Event)

        event_types = tuple((event_type,))
        return self.__wait_for_any_event(
            event_types,
            after_seq=0,
            until=until,
            timeout=timeout,
            timeout_msg=timeout_msg,
        )

    def wait_for_event(
        self,
        event_type: Type[AnyEvent],
        *,
        until: Optional[Callable[[AnyEvent], bool]] = None,
        after: Event | Response,
        timeout: Optional[float] = None,
        timeout_msg: Optional[str] = None,
    ) -> AnyEvent:
        """Wait for the next event of `event_type` after a given message.

        Search from "after some prior message" avoids races where
        the event has already been observed: a test can capture a response
        or event, run some action, and then wait for the *next* event of a
        given kind without matching against anything already in the log.

        Args:
            event_type: Event subclass to match.
            until: Optional predicate applied to each candidate. Only
                events for which `until(event)` is true are accepted.
            after: The prior event or response. Only events whose `seq`
                is strictly greater are considered.
            timeout: Override the history's default timeout, in seconds.
            timeout_msg: Extra context appended to the `TimeoutError`
                message if the wait times out.

        Returns:
            The first matching event after `after`.

        Raises:
            TimeoutError: If no matching event arrives within `timeout`.
            DAPError: If the history is closed before a match is found.
        """
        assert issubclass(event_type, Event)

        event_types = tuple((event_type,))
        return self.wait_for_any_event(
            event_types,
            after=after,
            until=until,
            timeout=timeout,
            timeout_msg=timeout_msg,
        )

    def wait_for_any_event(
        self,
        event_types: Tuple[Type[AnyEvent], ...],
        *,
        until: Optional[Callable[[AnyEvent], bool]] = None,
        after: Event | Response,
        timeout: Optional[float] = None,
        timeout_msg: Optional[str] = None,
    ):
        """Wait for the next event matching any of several types.

        Same semantics as `wait_for_event`, but the returned event may be
        an instance of any of the given `event_types`. Useful when a test
        is expecting the first of two different events.
        """
        assert after.type in (
            MessageType.EVENT,
            MessageType.RESPONSE,
        ), f"expects instance of 'Event' or 'Response' got {after}."
        return self.__wait_for_any_event(
            event_types,
            after_seq=after.seq,
            until=until,
            timeout=timeout,
            timeout_msg=timeout_msg,
        )

    def __wait_for_any_event(
        self,
        event_types: Tuple[Type[AnyEvent], ...],
        *,
        after_seq: int,
        until: Optional[Callable[[AnyEvent], bool]] = None,
        timeout: Optional[float] = None,
        timeout_msg: Optional[str] = None,
    ):
        assert after_seq >= 0, "response or event sequence must be greater than 0."
        assert isinstance(event_types, tuple), "expected a tuple of events."
        assert len(event_types) > 0, "expected at least one event to wait for."

        def make_error_msg(is_timeout: bool = True):
            event_names = [x.__name__ for x in event_types]
            prefix = f"Timed out after {timeout}s" if is_timeout else "Error while"
            err_msg = f"{prefix} waiting for any event that matches: {event_names}"
            err_msg += f" after sequence: {after_seq}."

            if timeout_msg:
                err_msg += f"\n\t{timeout_msg}."

            with self._new_event_condition:
                last_event = self._events[-1] if self._events else None
            err_msg += f"\n\tlast seen event: {last_event}."
            return err_msg

        def is_event_and_matches_condition(evt: Event):
            if not isinstance(evt, event_types):
                return False

            if until is None:
                return True

            matches = until(evt)
            return matches

        timeout = timeout or self._timeout
        try:
            event = self.__wait_until(
                is_event_and_matches_condition, after_seq=after_seq, timeout=timeout
            )
        except DAPError as err:
            # Add extra context to the error.
            err.args = (f"{err.args[0]}\n\t{make_error_msg(False)}", *err.args[1:])
            raise

        if event is None:
            raise TimeoutError(make_error_msg())

        # Sanity check.
        assert isinstance(event, event_types)
        return event

    def last_event(self) -> Event:
        with self._new_event_condition:
            if self._events:
                return self._events[-1]

        # There is no event with the sequence 0.
        # This exist in order provide the last event since the event list is empty.
        anchor_event = Event(seq=0, type=MessageType.EVENT, event="anchor_first_event")
        return anchor_event

    def __wait_until(
        self,
        matches_condition: Callable[[Event], bool],
        *,
        after_seq: int,
        timeout: float,
    ):
        """Waits until the `matches_condition` returns true for an exiting
        event or an incoming event. If the history is closed during the wait,
        raise a DAPError."""

        end_time = time.monotonic() + timeout
        start_idx = 0

        with self._new_event_condition:
            while True:
                seq_len = len(self._sequences)
                idx = bisect.bisect_right(self._sequences, after_seq, lo=start_idx)

                # Scan forward until we find a matching type.
                for event in itertools.islice(self._events, idx, seq_len):
                    if matches_condition(event):
                        return event
                start_idx = seq_len

                if self._is_closed:  # Can no longer receive new messages.
                    reason = self._closed_reason
                    last_evt = self._events[-1] if self._events else None
                    raise DAPError.history_closed(reason, last_evt) from reason

                remaining_time = end_time - time.monotonic()
                if remaining_time <= 0:
                    return None
                self._new_event_condition.wait(remaining_time)


class OutputBuffer:
    """A Thread safe io.StringIO."""

    def __init__(self):
        self._buf = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            return self._buf.write(text)

    def flush(self):
        with self._lock:
            self._buf.flush()

    def getvalue(self) -> str:
        with self._lock:
            return self._buf.getvalue()


def redirect_stream(
    in_stream: IO[bytes], out_buffer: OutputBuffer, thread_name: str
) -> threading.Thread:
    """
    Creates a new thread that redirects stream from `in_stream` to
    `out_stream`. We use this for the 'runInTerminal' process to send stdio
    to the session's output.

    Returns a thread that redirects the stream.
    """

    def read_loop(in_stream: IO[bytes], out_buffer: OutputBuffer):
        with contextlib.suppress(OSError, ValueError):  # Nothing to report.
            while True:
                chunk = in_stream.read(4096)
                if not chunk:
                    break

                out_buffer.write(chunk.decode(errors="replace"))
                out_buffer.flush()

    thread_name = f"redirect_{thread_name}"
    redirect_thread = threading.Thread(
        target=read_loop,
        name=thread_name,
        args=[in_stream, out_buffer],
        daemon=True,
    )
    redirect_thread.start()

    return redirect_thread


@runtime_checkable
class Transport(Protocol):
    """Interface representing a bidirectional transport.

    Implementations:
        `_StdioTransport`: speaks to the adapter using a subprocess's stdin/stdout.
            Used when the adapter is spawned as a child process.
        `_SocketTransport`: speaks to the adapter using socket. Used when the
            adapter is already running and exposes connection URI.
    """

    def write(self, data: bytes):
        ...

    def read(self, n: int) -> bytes:
        ...

    def readline(self) -> bytes:
        ...

    def close(self):
        """Close the transport.

        Buffered data will be flushed and transport closed.
        """
        ...

    @property
    def is_alive(self) -> bool:
        """Whether send or receive bytes through the transport."""
        ...


@dataclass(frozen=True)
class MessageHandler:
    on_response: Callable[[RawMessage], None]
    on_event: Callable[[RawMessage], None]
    on_reverse_request: Callable[[RawMessage], None]
    on_close: Optional[Callable[[Optional[Exception]], None]] = lambda _: None


class DAPConnection:
    """Manages the connection between a debug session and a debug adapter.

    Encodes and decodes messages using the DAP protocol, mapping them to
    dictionaries representing DAP types. Handles bidirectional communication
    between the session and the adapter, including error handling for
    failures from the debug adapter.
    """

    def __init__(self, connection_id: str, transport: Transport):
        assert isinstance(transport, Transport)
        self.id: str = connection_id
        self._transport = transport

        # A request that's been sent and is awaiting its response.
        self._in_flight_requests: dict[int, tuple[RawMessage, Future[RawMessage]]] = {}
        self._in_flight_lock = threading.Lock()
        # received_messages is not accessed anywhere. It only exists for debugging purposes.
        self._received_messages: list[RawMessage] = []

        # Event to sync when the Connection start listening for messages.
        self._is_ready = threading.Event()
        self._is_ready.clear()

    def start(self, handler: MessageHandler):
        self._read_loop(handler)

    def stop(self):
        if self._transport.is_alive:
            self._transport.close()

    @staticmethod
    def encode_message(message: dict):
        content = json.dumps(message, separators=(",", ":"))
        header = f"Content-Length: {len(content)}\r\n\r\n"
        data = f"{header}{content}".encode("utf-8")
        return data

    def send_request(self, request: Request) -> Future[RawMessage]:
        seq = request.seq
        response_future: Future[RawMessage] = Future()
        request_dict = request.to_dict()
        with self._in_flight_lock:
            self._in_flight_requests[seq] = (request_dict, response_future)
        self.send_message(request_dict)
        return response_future

    def send_message(self, message: dict):
        assert self.is_alive(), f"'{type(self).__name__}' is not running"
        data = DAPConnection.encode_message(message)
        self._transport.write(data)

    def is_alive(self):
        return self._transport.is_alive

    def wait_until_alive(self, timeout: float):
        return self._is_ready.wait(timeout)

    def _read_loop(self, handler: MessageHandler):
        self._is_ready.set()
        error = None
        try:
            while self.is_alive():
                message = DAPConnection.read_message(self._transport)
                if not message:
                    break

                self._received_messages.append(message)
                self._on_message(message, handler)

        except Exception as e:
            error = e
        finally:
            # Reject any unresolved requests so the test thread don't wait the
            # full timeout when the adapter exits or gets killed.
            with self._in_flight_lock:
                pending_futures = [f for _, f in self._in_flight_requests.values()]
                self._in_flight_requests = {}

            resp_error = error or DAPError("DAP connection closed before response.")
            for future in pending_futures:
                if not future.done():
                    future.set_exception(resp_error)

            with contextlib.suppress(Exception):
                self.stop()
                if on_close := handler.on_close:
                    on_close(error)

    def _on_message(self, message: RawMessage, handler: MessageHandler):
        msg_type = message.get("type")
        if msg_type == "response":
            request_seq = message["request_seq"]
            with self._in_flight_lock:
                in_flight = self._in_flight_requests.pop(request_seq, None)
            if in_flight is not None:
                request, response_future = in_flight
                if request["command"] == message["command"]:
                    response_future.set_result(message)
                else:
                    response_future.set_exception(
                        ValueError(
                            f"command mismatch in response"
                            f"{request['command']} != {message['command']}"
                        )
                    )
            handler.on_response(message)

        elif msg_type == "event":
            handler.on_event(message)

        elif msg_type == "request":
            handler.on_reverse_request(message)

        else:
            raise DAPError(f"Unknown message type: {msg_type}")

    @staticmethod
    def read_message(transport: Transport):
        HEADER_TERMINATOR = b"\r\n\r\n"
        CONTENT_LEN_PREFIX = b"Content-Length: "
        buffer = bytearray()

        while True:
            chunk = transport.readline()
            if not chunk:
                if buffer:
                    raise EOFError(f"unexpected EOF when parsing header: {buffer}")
                else:
                    return None
            buffer += chunk

            header_end = buffer.find(HEADER_TERMINATOR)
            if header_end == -1:
                continue
            header = buffer[:header_end]

            # Look for the Content-Length header.
            content_length = 0
            for line in header.split(b"\r\n"):
                if line.startswith(CONTENT_LEN_PREFIX):
                    content_length = int(line[len(CONTENT_LEN_PREFIX) :])
                    break
            else:
                raise DAPError(f"Invalid header: {header}")

            # Parse Content-Part.
            message_start = header_end + len(HEADER_TERMINATOR)
            buffer = buffer[message_start:]
            while len(buffer) < content_length:
                chunk = transport.read(content_length - len(buffer))
                if not chunk:
                    raise EOFError(f"unexpected EOF when parsing message: {buffer}")
                buffer += chunk

            message = json.loads(buffer.decode("utf-8"))
            return message


class _StdioTransport:
    def __init__(self, process: subprocess.Popen[bytes]):
        self._process = process

        stdin = self._process.stdin
        stdout = self._process.stdout
        assert stdin is not None
        assert stdout is not None
        self._stdin = stdin
        self._stdout = stdout

        self._is_closed = False
        assert self.is_alive

    def write(self, data: bytes):
        self._stdin.write(data)
        self._stdin.flush()

    def read(self, n: int) -> bytes:
        return self._stdout.read(n)

    def readline(self):
        return self._stdout.readline()

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        # Close stdin only. In Python3.8, closing stdout from main thread while the
        # reader thread is inside BufferedReader.read() will crash the interpreter.
        # The stdout cleanup happens via DebugAdapter.kill() when we kill the process
        with contextlib.suppress(OSError, ValueError):
            self._stdin.flush()
            self._stdin.close()

    @property
    def is_alive(self):
        if self._is_closed:
            return False

        # On Unix-like systems, attaching a debugger via ptrace temporarily reparents
        # the target process. This breaks Python's internal waitpid() tracking, causing
        # Popen.poll() to falsely return 0. Python assumes the missing child process
        # has already terminated cleanly, which leads to inaccurate status checks.
        return not self._stdout.closed and not self._stdin.closed


class _SocketTransport:
    def __init__(self, uri: str):
        self.uri = uri
        scheme, address = self.uri.split("://")
        if scheme == "unix-connect":  # unix-connect:///path
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(address)
        elif scheme == "connection":  # connection://[host]:port
            host, port = address.rsplit(":", 1)
            # create_connection with try both ipv4 and ipv6.
            self._socket = socket.create_connection((host.strip("[]"), int(port)))
        else:
            raise ValueError(f"invalid URI '{self.uri}' for socket")

        self._reader = self._socket.makefile("rb", buffering=-1)
        self._writer = self._socket.makefile("wb", buffering=-1)

        self._is_closed = False
        assert self.is_alive

    def write(self, data: bytes):
        self._writer.write(data)
        self._writer.flush()

    def read(self, n: int) -> bytes:
        return self._reader.read(n)

    def readline(self):
        return self._reader.readline()

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True

        with contextlib.suppress(OSError, ValueError):
            self._writer.flush()

        if self._socket.fileno() != -1:
            self._socket.shutdown(socket.SHUT_RDWR)

        self._writer.close()
        self._socket.close()

    @property
    def is_alive(self):
        if self._is_closed:
            return False
        try:
            _ = self._socket.getpeername()
        except socket.error:
            return False
        return True
