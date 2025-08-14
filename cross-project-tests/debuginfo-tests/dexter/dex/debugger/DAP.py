# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interface for communicating with a debugger via the DAP.
"""

import abc
from collections import defaultdict
import copy
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from enum import Enum

from dex.debugger.DebuggerBase import DebuggerBase, watch_is_active
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import StackFrame, SourceLocation, ProgramState
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode
from dex.utils.Logging import Logger
from dex.utils.Timeout import Timeout


# Helper enum used for colorizing DAP Message Log output.
class Color(Enum):
    CYAN = 36
    GREEN = 32
    YELLOW = 33
    RED = 31
    MAGENTA = 35

    def apply(self, text: str) -> str:
        return f"\033[{self.value}m{text}\033[0m"


class DAPMessageLogger:
    def __init__(self, context):
        self.dexter_logger = context.logger
        self.log_file: str = context.options.dap_message_log
        self.colorized: bool = context.options.colorize_dap_log
        self.indent = 2 if context.options.format_dap_log == "pretty" else None
        self.prefix_send: str = "->"
        self.prefix_recv: str = "<-"
        self.out_handle = None
        self.open = False
        self.lock = threading.Lock()

    def _custom_enter(self):
        self.open = True
        if self.log_file is None:
            return
        if self.log_file == "-":
            self.out_handle = sys.stdout
            return
        self.out_handle = open(self.log_file, "w+", encoding="utf-8")

    def _custom_exit(self):
        if self.out_handle is not None and self.log_file != "-":
            self.out_handle.close()
        self.open = False

    def _colorize_dap_message(self, message: dict) -> dict:
        if not self.colorized:
            return message
        colorized_message = copy.deepcopy(message)
        if colorized_message["type"] == "event":
            colorized_message["type"] = Color.YELLOW.apply("event")
            colorized_message["event"] = Color.YELLOW.apply(colorized_message["event"])
        elif colorized_message["type"] == "response":
            colorized_message["type"] = Color.GREEN.apply("response")
            colorized_message["command"] = Color.YELLOW.apply(
                colorized_message["command"]
            )
        elif colorized_message["type"] == "request":
            colorized_message["type"] = Color.CYAN.apply("request")
            colorized_message["command"] = Color.YELLOW.apply(
                colorized_message["command"]
            )
        return colorized_message

    def write_message(self, message: dict, incoming: bool):
        prefix = self.prefix_recv if incoming else self.prefix_send
        # ANSI escape codes get butchered by json.dumps(), so we fix them up here.
        message_str = json.dumps(
            self._colorize_dap_message(message), indent=self.indent
        ).replace("\\u001b", "\033")
        if self.out_handle is not None and self.open:
            with self.lock:
                self.out_handle.write(f"{prefix} {message_str}\n")
        elif not self.open:
            self.dexter_logger.warning(
                f'Attempted to write message after program closed: "{prefix} {message_str}"'
            )


# Debuggers communicate optional feature support.
class DAPDebuggerCapabilities:
    def __init__(self):
        self.supportsConfigurationDoneRequest: bool = False
        self.supportsFunctionBreakpoints: bool = False
        self.supportsConditionalBreakpoints: bool = False
        self.supportsHitConditionalBreakpoints: bool = False
        self.supportsEvaluateForHovers: bool = False
        self.supportsSetVariable: bool = False
        self.supportsStepInTargetsRequest: bool = False
        self.supportsModulesRequest: bool = False
        self.supportsValueFormattingOptions: bool = False
        self.supportsLogPoints: bool = False
        self.supportsSetExpression: bool = False
        self.supportsDataBreakpoints: bool = False
        self.supportsReadMemoryRequest: bool = False
        self.supportsWriteMemoryRequest: bool = False
        self.supportsDisassembleRequest: bool = False
        self.supportsCancelRequest: bool = False
        self.supportsSteppingGranularity: bool = False
        self.supportsInstructionBreakpoints: bool = False

    def update(self, logger: Logger, feature_dict: dict):
        for k, v in feature_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning(f"DAP: Unknown support flag: {k}")


# As DAP does not give us a trivially query-able process, we are responsible for maintaining our own state information,
# including what breakpoints are currently set, and whether the debugger is running or stopped.
# This class holds all state that is set based on events sent by the debug adapter; most responses are forwarded through
# to the main DAP class, though in a few cases where it is convenient for bookkeeping the DAPDebuggerState may read some
# information from the responses before forwarding them onwards.
class DAPDebuggerState:
    def __init__(self):
        ## Overall debugger state information.
        #
        # Whether we have received the initialize update yet.
        self.initialized: bool = False
        # Whether the debugger has successfully launched yet.
        self.launched: bool = False
        # The thread that we are debugging.
        # TODO: This is primitively handled right now, assuming that we only ever have one thread; if we want
        # support for debugging any multi-threaded program then we will need to track some more complex state.
        self.thread = None
        # True if the debuggee is currently executing.
        self.is_running: bool = False
        # True if the debuggee has finished executing.
        self.is_finished: bool = False

        ## Information for the program at a particular stopped point, which will be invalidated when execution resumes.
        #
        # Either None if the debuggee is currently running, or a string specifying the reason why the
        # debuggee is currently stopped otherwise.
        self.stopped_reason = None
        # If we were stopped for the reason 'breakpoint', this will contain a list of the DAP breakpoint IDs
        # responsible for stopping us.
        self.stopped_bps = []
        # For a currently stopped process, stores the mapping of frame indices (top of stack=0) to frameIds returned
        # from the debug adapter.
        self.frame_map = []

        # We use responses[idx] to refer to the response for the request sent with seq=idx, where the value
        # is either the response payload, or None if the response hasn't arrived yet.
        # Since requests are indexed from 1, we insert a 'None' at the front to ensure that the first real
        # entry is indexed correctly.
        self.responses = [None]
        # Map of DAP breakpoint IDs to resolved instruction addresses.
        self.bp_addr_map = {}

        # DAP features supported by the debugger.
        self.capabilities = DAPDebuggerCapabilities()

    def set_response(self, req_id: int, response: dict):
        if len(self.responses) > req_id:
            self.responses[req_id] = response
            return
        while len(self.responses) < req_id:
            self.responses.append(None)
        self.responses.append(response)

    # As the receiver thread does not know when a request has been sent, and only the receiver thread should write to the DebuggerState object,
    # the responses list may not have been populated with a None for a pending request at the time that the main thread expects it. Therefore,
    # we use this getter to account for requests that the receiver thread is unaware of.
    def get_response(self, req_id: int):
        if len(self.responses) <= req_id:
            return None
        return self.responses[req_id]


# DAP Communication model:
# - Communication is message-based, not stateful - we cannot simply query information from the debugger as we can with
#   other debugger implementations, we need to maintain local state.
# - All messages are utf-encoded JSON, which we convert to/from python dicts via methods above; some amount of
#   bookkeeping is performed automatically in the DAP class.
# - Commands and queries are sent via 'request' messages, for which a corresponding 'response' will always be sent back
#   by the adapter indicating success/failure, containing data related to the request.
# - The adapter will also send 'event' messages, indicating state changes in the debugger - for example, when the
#   debugger has stopped at a breakpoint.
# In order to handle this, we run a separate thread that will continuously insert any messages received
# from the adapter into a queue, which the main thread will read; generally, our response to any read message
# is to update our state, which Dexter's DebuggerController will then read.
class DAP(DebuggerBase, metaclass=abc.ABCMeta):
    def __init__(self, context, *args):
        self._debugger_state = DAPDebuggerState()
        self._proc = None
        self._receiver_thread = None
        self._err_thread = None
        self.seq = 0
        self.target_proc_id = -1
        self.max_bp_id = 0
        # Mapping of active breakpoints per-file - intentionally excludes breakpoints that we have deleted.
        # { file -> [dex_breakpoint_id]}
        self.file_to_bp = defaultdict(list)
        # { dex_breakpoint_id -> (file, line, condition) }
        self.bp_info = {}
        # We don't rely on IDs returned directly from the debug adapter. Instead, we use dexter breakpoint IDs, and
        # maintain a two-way-mapping of dex_bp_id<->dap_bp_id. This also allows us to defer the setting of breakpoints
        # in the debug adapter itself until necessary.
        # NB: The debug adapter may merge dexter-side breakpoints into a single debugger-side breakpoint; therefore, the
        # DAP->Dex mapping is one-to-many.
        self.dex_id_to_dap_id = {}
        self.dap_id_to_dex_ids = {}
        self.pending_breakpoints: bool = False
        # List of breakpoints, indexed by BP ID
        # Each entry has the source file (for use in referencing desired_bps), and the DA-assigned
        # ID for that breakpoint if it has one (if it has been removed or not yet created then it will be None).
        # self.bp_source_list: list[(str, int)]
        self.message_logger = None
        super(DAP, self).__init__(context, *args)

    @property
    @abc.abstractmethod
    def _debug_adapter_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _debug_adapter_executable(self) -> str:
        pass

    @property
    def _debug_adapter_launch_args(self) -> list:
        return []

    @staticmethod
    def make_request(command: str, arguments=None) -> dict:
        request = {"type": "request", "command": command}
        if arguments is not None:
            request["arguments"] = arguments
        return request

    @staticmethod
    def make_initialize_request(adapterID: str) -> dict:
        return DAP.make_request(
            "initialize",
            {
                "clientID": "dexter",
                "adapterID": adapterID,
                "pathFormat": "path",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "supportsVariableType": True,
                "supportsVariablePaging": True,
                "supportsRunInTerminalRequest": False,
            },
        )

    class BreakpointRequest:
        def __init__(self, line: int, condition=None):
            self.line = line
            self.condition = condition

        def toDict(self) -> dict:
            result = {"line": self.line}
            if self.condition is not None:
                result["condition"] = self.condition
            return result

    @staticmethod
    def make_set_breakpoint_request(source: str, bps) -> dict:
        return DAP.make_request(
            "setBreakpoints",
            {"source": {"path": source}, "breakpoints": [bp.toDict() for bp in bps]},
        )

    ############################################################################
    ## DAP communication & state-handling functions

    # Sends a request to the adapter, returning the seq value of the request.
    def send_message(self, payload: dict) -> int:
        self.seq = self.seq + 1
        payload["seq"] = self.seq
        self.message_logger.write_message(payload, False)
        body = json.dumps(payload)
        message = f"Content-Length: {len(body)}\r\n\r\n{body}".encode("utf-8")
        self._proc.stdin.write(message)
        self._proc.stdin.flush()
        return self.seq

    def _handle_message(
        message: dict, debugger_state: DAPDebuggerState, logger: Logger
    ):
        # We only support events and responses, we do not implement any reverse-requests.
        # TODO: If we find cases where 'seq' becomes important, we need to read it here and process
        # pending messages in order.
        if message["type"] == "event":
            event_type = message["event"]
            event_details = message.get("body")
            if event_type == "initialized":
                debugger_state.initialized = True
            elif event_type == "process":
                debugger_state.launched = True
                debugger_state.is_running = True
            # The debugger has stopped for some reason.
            elif event_type == "stopped":
                stop_reason = event_details["reason"]
                debugger_state.is_running = False
                debugger_state.stopped_reason = stop_reason
                debugger_state.stopped_bps = event_details.get("hitBreakpointIds", [])
                debugger_state.thread = event_details["threadId"]
            elif event_type == "breakpoint":
                # We handle most BP information in the main DAP thread by reading responses to breakpoint requests;
                # some information is only passed via event, however, which we store here.
                breakpoint_details = event_details["breakpoint"]
                if "instructionReference" in breakpoint_details:
                    debugger_state.bp_addr_map[
                        breakpoint_details["id"]
                    ] = breakpoint_details["instructionReference"]
            elif event_type == "exited" or event_type == "terminated":
                debugger_state.stopped_reason = event_type
                debugger_state.is_running = False
                debugger_state.is_finished = True
            # We may receive this event before or after the response to the corresponding "continue" request.
            elif event_type == "continued":
                debugger_state.is_running = True
                # Reset all state that is invalidated upon program continue.
                debugger_state.stopped_reason = None
                debugger_state.stopped_bps = []
                debugger_state.frame_map = []
            elif event_type == "thread":
                if (
                    event_details["reason"] == "started"
                    and debugger_state.thread is None
                ):
                    debugger_state.thread = event_details["threadId"]
            elif event_type == "capabilities":
                # Unchanged capabilites may not be included.
                debugger_state.capabilities.update(logger, event_details)
            # There are many events we do not care about, just skip processing them.
            else:
                pass
        elif message["type"] == "response":
            request_seq = message["request_seq"]
            debugger_state.set_response(request_seq, message)
            # TODO: We also receive a "continued" event, but it seems reasonable to set state based on either the
            # response or the event, since the DAP does not specify an order in which they are sent. May need revisiting
            # if there turns out to be some odd ordering issues, e.g. if we can receive messages in the order
            # ["response: continued", "event: stopped", "event: continued"].
            if (
                message["command"] in ["continue", "stepIn", "next", "stepOut"]
                and message["success"] == True
            ):
                debugger_state.is_running = True
                # Reset all state that is invalidated upon program continue.
                debugger_state.stopped_reason = None
                debugger_state.stopped_bps = []
                debugger_state.frame_map = []
            # It is useful to cache a mapping of frames; since this is invalidated when we continue, and only this
            # message-handling thread should write to debugger_state, we do so while handling the response for
            # convenience.
            if message["command"] == "stackTrace" and message["success"] == True:
                debugger_state.frame_map = [
                    stackframe["id"] for stackframe in message["body"]["stackFrames"]
                ]
            # The debugger communicates which optional DAP features are
            # supported in its initalize response.
            if message["command"] == "initialize" and message["success"] == True:
                body = message.get("body")
                if body:
                    debugger_state.capabilities.update(logger, body)

    def _colorize_dap_message(message: dict) -> dict:
        colorized_message = copy.deepcopy(message)
        if colorized_message["type"] == "event":
            colorized_message["type"] = "<y>event</>"
            colorized_message["event"] = f"<y>{colorized_message['event']}</>"
        elif colorized_message["type"] == "response":
            colorized_message["type"] = "<g>response</>"
            colorized_message["command"] = f"<y>{colorized_message['command']}</>"
        elif colorized_message["type"] == "request":
            colorized_message["type"] = "<b>request</>"
            colorized_message["command"] = f"<y>{colorized_message['command']}</>"
        return colorized_message

    def _read_dap_output(
        proc: subprocess.Popen,
        debugger_state: DAPDebuggerState,
        message_logger: DAPMessageLogger,
        logger: Logger,
    ):
        buffer: bytes = b""
        while True:
            chunk: bytes = proc.stdout.read(1)
            if not chunk:
                break
            buffer += chunk
            if b"\r\n\r\n" in buffer:
                header, rest = buffer.split(b"\r\n\r\n", 1)
                content_length = int(header.decode().split(":")[1].strip())
                while len(rest) < content_length:
                    rest += proc.stdout.read(content_length - len(rest))
                message = json.loads(rest[:content_length])
                message_logger.write_message(message, True)
                DAP._handle_message(message, debugger_state, logger)
                buffer = rest[content_length:]

    def _read_dap_err(proc: subprocess.Popen, logger: Logger):
        while True:
            err: bytes = proc.stderr.readline()
            if len(err) > 0:
                logger.error(f"DAP server: {err.decode().strip()}")

    def _custom_init(self):
        self.context.logger.note(
            f"Opening DAP server: {shlex.join([self._debug_adapter_executable] + self._debug_adapter_launch_args)}"
        )
        self.message_logger = DAPMessageLogger(self.context)
        self.message_logger._custom_enter()
        self._proc = subprocess.Popen(
            [self._debug_adapter_executable] + self._debug_adapter_launch_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._receiver_thread = threading.Thread(
            target=DAP._read_dap_output,
            args=(
                self._proc,
                self._debugger_state,
                self.message_logger,
                self.context.logger,
            ),
            daemon=True,
        )
        self._err_thread = threading.Thread(
            target=DAP._read_dap_err,
            args=(self._proc, self.context.logger),
            daemon=True,
        )
        self._receiver_thread.start()
        self._err_thread.start()
        init_req = self.send_message(
            self.make_initialize_request(self._debug_adapter_name)
        )
        assert self._proc.poll() is None, "Process has closed unexpectedly early?"
        self._await_response(init_req)

    def _custom_exit(self):
        if self._proc is not None:
            dc_req = self.send_message(self.make_request("disconnect"))
            dc_req_timeout = 3
            try:
                result = self._await_response(dc_req, dc_req_timeout)
                if not result["success"]:
                    self.context.logger.warning(
                        "The disconnect request sent to the DAP server failed; forcibly shutting down DAP server."
                    )
                else:
                    self.context.logger.note(
                        "Successfully disconnected from DAP server."
                    )
            except:
                # We're going to kill the process regardless, we just want to give the target a chance to shut down
                # gracefully first.
                self.context.logger.warning(
                    f"The disconnect request sent to the DAP server timed out after {dc_req_timeout}s; forcibly shutting down DAP server."
                )
                pass
            self._proc.kill()
            self._proc = None
        self.message_logger._custom_exit()

    # Waits for a response to the request with the given seq, optionally raising an error
    # if the response takes too long (blocks forever by default/if timeout=0).
    def _await_response(self, seq: int, timeout: float = 0.0) -> dict:
        timeout_check = Timeout(timeout)
        while self._debugger_state.get_response(seq) is None:
            if timeout_check.timed_out():
                if self._proc.poll() is not None:
                    self.context.logger.error(
                        f"Debug adapter exited while Dexter is awaiting response? Result: {self._proc.poll()}"
                    )
                raise TimeoutError(
                    f"Timed out while waiting for response to DAP request {seq}"
                )
            time.sleep(0.001)
        return self._debugger_state.get_response(seq)

    ## End of DAP communication methods
    ############################################################################

    def _translate_stop_reason(self, reason):
        if reason is None:
            return None
        if "breakpoint" in reason:
            return StopReason.BREAKPOINT
        if reason == "step":
            return StopReason.STEP
        if reason == "exited" or reason == "terminated":
            return StopReason.PROGRAM_EXIT
        if reason == "exception":
            return StopReason.ERROR
        return StopReason.OTHER

    def _load_interface(self):
        if not os.path.isfile(self._debug_adapter_executable):
            raise LoadDebuggerException(
                f'debug adapter "{self._debug_adapter_executable}" does not exist',
                sys.exc_info(),
            )
        # We don't make use of _interface, so return nothing.

    @property
    @abc.abstractmethod
    def version(self):
        """The version of this DAP debugger."""

    ############################################################################
    ## Breakpoint Methods

    def get_next_bp_id(self):
        new_id = self.max_bp_id
        self.max_bp_id += 1
        return new_id

    def get_current_bps(self, source):
        if source in self.file_to_bp:
            return self.file_to_bp[source]
        return []

    def _update_requested_bp_list(self, bp_list):
        """Can be overridden for any specific implementations that need further processing before sending breakpoints to
        the debug adapter, e.g. in LLDB we cannot store multiple breakpoints at a single location, and therefore must
        combine conditions for breakpoints at the same location."""
        return bp_list

    # For a source file, returns the list of BreakpointRequests for the breakpoints in that file, which can be sent to
    # the debug adapter.
    def _get_desired_bps(self, file: str):
        bp_list = [
            DAP.BreakpointRequest(line, cond)
            for (_, line, cond) in map(
                lambda dex_bp_id: self.bp_info[dex_bp_id], self.get_current_bps(file)
            )
        ]
        return self._update_requested_bp_list(bp_list)

    def clear_breakpoints(self):
        # We don't actually need to do anything here - even if breakpoints were preserved between runs, we will
        # automatically clear old breakpoints on the first 'setBreakpoints' message.
        pass

    def _add_breakpoint(self, file, line):
        return self._add_conditional_breakpoint(file, line, None)

    def _add_conditional_breakpoint(self, file, line, condition):
        new_id = self.get_next_bp_id()
        self.file_to_bp[file].append(new_id)
        self.bp_info[new_id] = (file, line, condition)
        self.pending_breakpoints = True
        return new_id

    def _flush_breakpoints(self):
        if not self.pending_breakpoints:
            return
        for file in self.file_to_bp.keys():
            desired_bps = self._get_desired_bps(file)
            request_id = self.send_message(
                self.make_set_breakpoint_request(file, desired_bps)
            )
            result = self._await_response(request_id, 10)
            if not result["success"]:
                raise DebuggerException(f"could not set breakpoints for '{file}'")
            # The debug adapter may have chosen to merge our breakpoints. From here we need to identify such cases and
            # handle them so that our internal bookkeeping is correct.
            dex_bp_ids = self.get_current_bps(file)
            dap_bp_ids = [bp["id"] for bp in result["body"]["breakpoints"]]
            if len(dex_bp_ids) != len(dap_bp_ids):
                self.context.logger.error(
                    f"Sent request to set {len(dex_bp_ids)} breakpoints, but received {len(dap_bp_ids)} in response."
                )
            visited_dap_ids = set()
            for i, dex_bp_id in enumerate(dex_bp_ids):
                dap_bp_id = dap_bp_ids[i]
                self.dex_id_to_dap_id[dex_bp_id] = dap_bp_id
                # We take the mappings in the response as the canonical mapping, meaning that if the debug server has
                # simply *changed* the DAP ID for a breakpoint we overwrite the existing mapping rather than adding to
                # it, but if we receive the same DAP ID for multiple Dex IDs *then* we store a one-to-many mapping.
                if dap_bp_id in visited_dap_ids:
                    self.dap_id_to_dex_ids[dap_bp_id].append(dex_bp_id)
                else:
                    self.dap_id_to_dex_ids[dap_bp_id] = [dex_bp_id]
                    visited_dap_ids.add(dap_bp_id)
        self.pending_breakpoints = False

    def _confirm_triggered_breakpoint_ids(self, dex_bp_ids):
        """Can be overridden for any specific implementations that need further processing from the debug server's
        reported 'hitBreakpointIds', e.g. in LLDB where we the ID for every breakpoint at the current PC, even if some
        are conditional and their condition is not met."""
        return dex_bp_ids

    def get_triggered_breakpoint_ids(self):
        # Breakpoints can only have been triggered if we've hit one.
        stop_reason = self._translate_stop_reason(self._debugger_state.stopped_reason)
        if stop_reason != StopReason.BREAKPOINT:
            return []
        breakpoint_ids = set(
            [
                dex_id
                for dap_id in self._debugger_state.stopped_bps
                for dex_id in self.dap_id_to_dex_ids[dap_id]
            ]
        )
        return self._confirm_triggered_breakpoint_ids(breakpoint_ids)

    def delete_breakpoints(self, ids):
        per_file_deletions = defaultdict(list)
        for dex_bp_id in ids:
            source, _, _ = self.bp_info[dex_bp_id]
            per_file_deletions[source].append(dex_bp_id)
        for file, deleted_ids in per_file_deletions.items():
            old_len = len(self.file_to_bp[file])
            self.file_to_bp[file] = [
                bp_id for bp_id in self.file_to_bp[file] if bp_id not in deleted_ids
            ]
            if len(self.file_to_bp[file]) != old_len:
                self.pending_breakpoints = True

    ## End of breakpoint methods
    ############################################################################

    @classmethod
    @abc.abstractmethod
    def _get_launch_params(self, cmdline):
        """ "Set the debugger-specific params used in a launch request."""

    def launch(self, cmdline):
        assert len(self.file_to_bp.keys()) > 0

        if self.context.options.target_run_args:
            cmdline += shlex.split(self.context.options.target_run_args)

        launch_request = self._get_launch_params(cmdline)

        # For some reason, we *must* submit in the order launch->configurationDone, and then we will receive responses
        # in the order configurationDone->launch.
        self._flush_breakpoints()
        launch_req_id = self.send_message(self.make_request("launch", launch_request))
        config_done_req_id = self.send_message(self.make_request("configurationDone"))
        config_done_response = self._await_response(config_done_req_id)
        assert config_done_response["success"], "Should simply receive an affirmative?"
        launch_response = self._await_response(launch_req_id)
        if not launch_response["success"]:
            raise DebuggerException(
                f"failure launching debugger: \"{launch_response['body']['error']['format']}\""
            )
        # We can't interact meaningfully with the process until we have the thread ID and confirmation that the process
        # has finished launching.
        while self._debugger_state.thread is None or not self._debugger_state.launched:
            time.sleep(0.001)

    # LLDB has unique stepping behaviour w.r.t. breakpoints that needs to be handled after completing a step, so we use
    # an overridable hook to enable debugger-specific behaviour.
    def _post_step_hook(self):
        """Hook to be executed after completing a step request."""

    def _step(self, step_request_string):
        self._flush_breakpoints()
        step_req_id = self.send_message(
            self.make_request(
                step_request_string, {"threadId": self._debugger_state.thread}
            )
        )
        response = self._await_response(step_req_id)
        if not response["success"]:
            raise DebuggerException(
                f"failed to perform debugger action: '{step_request_string}'"
            )
        # If we've "stepped" to a breakpoint, then continue to hit the breakpoint properly.
        # NB: This is an issue that only seems relevant to LLDB, but is also harmless outside of LLDB; if it turns out
        #     to cause issues for other debuggers, we can move it to a post-step hook.
        while self._debugger_state.is_running:
            time.sleep(0.001)
        self._post_step_hook()

    def step_in(self):
        self._step("stepIn")

    def step_next(self):
        self._step("next")

    def step_out(self):
        self._step("stepOut")

    def go(self) -> ReturnCode:
        self._flush_breakpoints()
        continue_req_id = self.send_message(
            self.make_request("continue", {"threadId": self._debugger_state.thread})
        )
        response = self._await_response(continue_req_id)
        if not response["success"]:
            raise DebuggerException("failed to continue")
        # Assuming the request to continue succeeded, we still need to wait to receive an event back from the debugger
        # indicating that we have successfully resumed.

    def _get_step_info(self, watches, step_index):
        assert (
            not self._debugger_state.is_running
        ), "Cannot get step info while debugger is running!"
        trace_req_id = self.send_message(
            self.make_request("stackTrace", {"threadId": self._debugger_state.thread})
        )
        trace_response = self._await_response(trace_req_id)
        if not trace_response["success"]:
            raise DebuggerException("failed to get stack frames")
        stackframes = trace_response["body"]["stackFrames"]

        frames = []
        state_frames = []

        for idx, stackframe in enumerate(stackframes):
            # FIXME: No source, skip the frame! Currently I've only observed this for frames below main, so we break
            # here; if it happens elsewhere, then this will break more stuff and we'll come up with a better solution.
            if (
                stackframe.get("source") is None
                or stackframe["source"].get("path") is None
            ):
                break
            loc_dict = {
                "path": stackframe["source"]["path"],
                "lineno": stackframe["line"],
                "column": stackframe["column"],
            }
            loc = LocIR(**loc_dict)
            valid_loc_for_watch = loc.path and os.path.exists(loc.path)
            frame = FrameIR(
                function=self._sanitize_function_name(stackframe["name"]),
                is_inlined=stackframe["name"].startswith("[Inline Frame]"),
                loc=loc,
            )

            # We skip frames that are below "main", since we do not expect those to be user code.
            fname = frame.function or ""  # pylint: disable=no-member
            if any(name in fname for name in self.frames_below_main):
                break

            frames.append(frame)

            state_frame = StackFrame(
                function=frame.function,
                is_inlined=frame.is_inlined,
                location=SourceLocation(**loc_dict),
                watches={},
            )
            if valid_loc_for_watch:
                for expr in map(
                    # Filter out watches that are not active in the current frame,
                    # and then evaluate all the active watches.
                    lambda watch_info, idx=idx: self.evaluate_expression(
                        watch_info.expression, idx
                    ),
                    filter(
                        lambda watch_info, idx=idx, line_no=loc.lineno, loc_path=loc.path: watch_is_active(
                            watch_info, loc_path, idx, line_no
                        ),
                        watches,
                    ),
                ):
                    state_frame.watches[expr.expression] = expr
            state_frames.append(state_frame)

        if len(frames) == 1 and frames[0].function is None:
            frames = []
            state_frames = []

        reason = self._translate_stop_reason(self._debugger_state.stopped_reason)

        return StepIR(
            step_index=step_index,
            frames=frames,
            stop_reason=reason,
            program_state=ProgramState(state_frames),
        )

    @property
    def is_running(self):
        return self._debugger_state.is_running

    @property
    def is_finished(self):
        return self._debugger_state.is_finished

    @property
    def frames_below_main(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _evaluate_result_value(expression: str, result_string: str) -> ValueIR:
        """For the result of an "evaluate" message, return a ValueIR. Implementation must be debugger-specific."""

    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        # The frame_idx passed in here needs to be translated to the debug adapter's internal frame ID.
        dap_frame_id = self._debugger_state.frame_map[frame_idx]
        eval_req_id = self.send_message(
            self.make_request(
                "evaluate",
                {
                    "expression": expression,
                    "frameId": dap_frame_id,
                    "context": "watch",
                },
            )
        )
        eval_response = self._await_response(eval_req_id)
        if not eval_response["success"]:
            result: str = eval_response["message"]
        else:
            result: str = eval_response["body"]["result"]
        type_str = eval_response["body"].get("type")

        return self._evaluate_result_value(expression, result, type_str)
