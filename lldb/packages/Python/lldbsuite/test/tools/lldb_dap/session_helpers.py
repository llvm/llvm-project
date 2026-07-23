# FIXME: remove when LLDB_MINIMUM_PYTHON_VERSION > 3.8
from __future__ import annotations

import base64
import dataclasses
import logging
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from .types import (
    AttachArgs,
    Breakpoint,
    BreakpointEvent,
    BreakpointLocationsArgs,
    CompletionsArgs,
    ConfigurationDoneArgs,
    ContinueArgs,
    ContinuedEvent,
    DataBreakpoint,
    DataBreakpointInfoArgs,
    DisassembleArgs,
    DisconnectArgs,
    EmptyBodyResponse,
    ErrorResponse,
    EvaluateArgs,
    EvaluateContext,
    EvaluateResponse,
    Event,
    EventName,
    ExceptionFilterOptions,
    ExceptionInfoArgs,
    ExceptionOptions,
    ExitedEvent,
    FunctionBreakpoint,
    InitializeArgs,
    InitializedEvent,
    InstructionBreakpoint,
    InvalidatedEvent,
    LaunchArgs,
    LocationsArgs,
    MemoryEvent,
    ModuleEvent,
    ModuleReason,
    ModulesArgs,
    NextArgs,
    OutputCategory,
    OutputEvent,
    ProcessEvent,
    ReadMemoryArgs,
    ReadMemoryResponse,
    Response,
    RestartArgs,
    Scope,
    ScopesArgs,
    SetBreakpointsArgs,
    SetDataBreakpointsArgs,
    SetExceptionBreakpointsArgs,
    SetFunctionBreakpointsArgs,
    SetInstructionBreakpointsArgs,
    SetVariableArgs,
    SetVariableResponse,
    Source,
    SourceBreakpoint,
    StackFrame,
    StackFrameFormat,
    StackTraceArgs,
    StepInArgs,
    StepOutArgs,
    SteppingGranularity,
    StoppedEvent,
    StoppedReason,
    TerminatedEvent,
    ThreadsArgs,
    ValueFormat,
    Variable,
    VariablePresentationHint,
    VariablesArgs,
    WriteMemoryArgs,
)
from .session import PendingResponse, Session
from .utils import DebugAdapter, SubProcessSpawner

T = TypeVar("T")


class ThreadContext:
    """Lazy view of a debug adapter thread.

    Thread ids do not have a limited lifetime, so this context is long-lived.
    It can be reused after continue and stepXXX requests.
    """

    def __init__(self, thread_id: int, session: DAPTestSession):
        self._thread_id: int = thread_id
        self._session: DAPTestSession = session

    @property
    def thread_id(self) -> int:
        return self._thread_id

    def step_in(
        self,
        *,
        targetId: Optional[int] = None,
        granularity: SteppingGranularity = "statement",
    ):
        return self._session.step_in(
            threadId=self.thread_id, targetId=targetId, granularity=granularity
        )

    def step_over(self, *, granularity: SteppingGranularity = "statement"):
        return self._session.step_over(threadId=self.thread_id, granularity=granularity)

    def step_out(self, *, granularity: SteppingGranularity = "statement"):
        return self._session.step_out(threadId=self.thread_id, granularity=granularity)

    def top_frame(
        self,
        *,
        format: Optional[StackFrameFormat] = None,
    ) -> FrameContext:
        return self.frames(levels=1, format=format)[0]

    def frames(
        self,
        *,
        startFrame: Optional[int] = None,
        levels: Optional[int] = None,
        format: Optional[StackFrameFormat] = None,
    ) -> list[FrameContext]:
        args = StackTraceArgs(
            self._thread_id, startFrame=startFrame, levels=levels, format=format
        )
        response = self._session.send_request(args).result()
        generation = self._session._current_stop_generation()
        return [
            FrameContext(frame, self._session, generation)
            for frame in response.body.stackFrames
        ]


class FrameContext:
    """Lazy view of a stack frame. Valid only within its stop generation."""

    def __init__(self, frame: StackFrame, session: DAPTestSession, generation: int):
        self._frame = frame
        self._session = session
        self._generation = generation
        self._scopes: Optional[list[ScopeContext]] = None

    @property
    def frame(self) -> StackFrame:
        self._session._check_stop_generation(self._generation, self)
        return self._frame

    @property
    def id(self) -> int:
        return self.frame.id

    @property
    def name(self) -> str:
        return self.frame.name

    def __dir__(self):
        # Hide the property fields that may call 'ScopesRequest' from the debugger.
        # The python debugger will hang because it is waiting for a response
        # when viewing the FrameContext.
        hidden = {"locals", "globals", "registers", "scopes"}
        return (attr for attr in super().__dir__() if attr not in hidden)

    def source_and_line(self) -> tuple[str, int]:
        frame = self.frame
        assert frame.source is not None
        assert frame.source.path is not None
        assert frame.line is not None
        return frame.source.path, frame.line

    def scopes(self) -> list[ScopeContext]:
        self._session._check_stop_generation(self._generation, self)
        if self._scopes is None:
            scope_args = ScopesArgs(frameId=self._frame.id)
            response = self._session.send_request(scope_args).result()
            self._scopes = [
                ScopeContext(scope, self._session, self._generation)
                for scope in response.body.scopes
            ]
        return self._scopes

    def scope(self, name: str) -> ScopeContext:
        scopes = self.scopes()
        for scope in scopes:
            if scope.scope.name == name:
                return scope
        scope_names = [scope.scope.name for scope in scopes]
        self._session.test_case.fail(
            f"scope '{name}' not in frame scopes: {scope_names}"
        )

    @property
    def locals(self) -> ScopeContext:
        return self.scope("Locals")

    @property
    def globals(self) -> ScopeContext:
        return self.scope("Globals")

    @property
    def registers(self) -> ScopeContext:
        return self.scope("Registers")

    def evaluate(
        self,
        expression: str,
        *,
        context: Optional[EvaluateContext] = None,
        format: Optional[ValueFormat] = None,
    ):
        """Evaluates `expression` in this frame's context."""
        self._session._check_stop_generation(self._generation, self)
        return self._session.evaluate(
            expression, frameId=self._frame.id, context=context, format=format
        )

    def disassemble(self):
        self._session._check_stop_generation(self._generation, self)

        mem_ref = self._frame.instructionPointerReference
        if mem_ref is None:
            self._session.test_case.fail(
                f"expects 'instructionPointerReference' for frame {self.frame}"
            )
        return self._session.disassemble(
            mem_ref, instructionOffset=0, instructionCount=100
        )


class _VariableContainer:
    """Shared dict-like behaviour for contexts that hold a variablesReference.

    The optional `_value_format` is passed into every child-fetching
    `variables` request in the container.
    A child `VariableContext` inherits its parent's format,
    so walking `locals.with_format(hex)["pt"]["x"]` keeps hex formatting all
    the way down without the caller repeating it at each step.
    """

    _session: DAPTestSession
    _generation: int
    _value_format: Optional[ValueFormat] = None

    def _fetch_variables(
        self,
        variables_reference: int,
        *,
        filter: Optional[Literal["indexed", "named"]] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
    ) -> list[VariableContext]:
        self._session._check_stop_generation(self._generation, self)
        variables = self._session.get_variables(
            variables_reference,
            filter=filter,
            start=start,
            count=count,
            format=self._value_format,
        )
        return [
            VariableContext(var, self._session, self._generation, self._value_format)
            for var in variables
        ]

    def page(
        self,
        *,
        filter: Optional[Literal["indexed", "named"]] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
    ) -> list[VariableContext]:
        """Fetch a subset of children with paging/filter arguments.

        Inherits the container's value format.
        """
        return self._fetch_variables(
            self._container_reference(),
            filter=filter,
            start=start,
            count=count,
        )

    def set(
        self, name: str, value, *, is_hex: bool = False
    ) -> SetVariableResponse | ErrorResponse:
        """Sends a `setVariable` request for a named child."""
        self._session._check_stop_generation(self._generation, self)
        return self._session.set_variable(
            name, value, variablesReference=self._container_reference(), is_hex=is_hex
        )

    def _container_reference(self) -> int:
        raise NotImplementedError

    def _by_name(self) -> dict[str, VariableContext]:
        return {child.name: child for child in self._children()}

    def _children(self) -> list[VariableContext]:
        raise NotImplementedError

    def __getitem__(self, name: str) -> VariableContext:
        by_name = self._by_name()
        try:
            return by_name[name]
        except KeyError:
            self._session.test_case.fail(
                f"'{name}' not found in {self}, has: {list(by_name)}"
            )

    def __contains__(self, name: object) -> bool:
        return name in self._by_name()

    def __iter__(self) -> Iterator[VariableContext]:
        return iter(self._children())

    def __len__(self) -> int:
        return len(self._children())

    def __str__(self) -> str:
        return type(self).__name__


class ScopeContext(_VariableContainer):
    """Lazy view of a scope's variables. Valid only within its stop generation."""

    def __init__(
        self,
        scope: Scope,
        session: DAPTestSession,
        generation: int,
        value_format: Optional[ValueFormat] = None,
    ):
        self._scope = scope
        self._session = session
        self._generation = generation
        self._value_format = value_format

    @property
    def scope(self) -> Scope:
        self._session._check_stop_generation(self._generation, self)
        return self._scope

    @property
    def name(self) -> str:
        return self.scope.name

    @property
    def variablesReference(self) -> int:
        return self.scope.variablesReference

    def variables(self) -> list[VariableContext]:
        return self._fetch_variables(self._scope.variablesReference)

    def with_format(self, *, is_hex: bool = False) -> ScopeContext:
        """Return a new ScopeContext that applies the ValueFormat."""
        value_format = ValueFormat(hex=True) if is_hex else None
        return ScopeContext(self._scope, self._session, self._generation, value_format)

    def _container_reference(self) -> int:
        return self._scope.variablesReference

    def _children(self) -> list[VariableContext]:
        return self.variables()

    def __str__(self) -> str:
        return f"scope '{self._scope.name}'"


class VariableContext(_VariableContainer):
    """Lazy view of a variable and (optionally) its children.

    Valid only within its' stop generation.
    """

    def __init__(
        self,
        variable: Variable,
        session: DAPTestSession,
        generation: int,
        value_format: Optional[ValueFormat] = None,
    ):
        self._variable = variable
        self._session = session
        self._generation = generation
        self._value_format = value_format

    @property
    def variable(self) -> Variable:
        self._session._check_stop_generation(self._generation, self)
        return self._variable

    @property
    def name(self) -> str:
        return self._variable.name

    @property
    def value(self) -> str:
        return self._variable.value

    @property
    def value_as_int(self) -> int:
        return self._variable.value_as_int

    @property
    def type(self) -> Optional[str]:
        return self._variable.type

    @property
    def variablesReference(self) -> int:
        return self._variable.variablesReference

    @property
    def memoryReference(self) -> Optional[str]:
        return self._variable.memoryReference

    @property
    def indexedVariables(self) -> Optional[int]:
        return self._variable.indexedVariables

    @property
    def namedVariables(self) -> Optional[int]:
        return self._variable.namedVariables

    @property
    def has_children(self) -> bool:
        return self._variable.variablesReference > 0

    def children(self) -> list[VariableContext]:
        if not self.has_children:
            self._session.test_case.fail(
                f"variable '{self._variable.name}' has no children"
            )
        return self._fetch_variables(self._variable.variablesReference)

    def with_format(self, *, is_hex: bool = False) -> VariableContext:
        """Return a new VariableContext that applies the ValueFormat"""
        value_format = ValueFormat(hex=True) if is_hex else None
        return VariableContext(
            self._variable, self._session, self._generation, value_format
        )

    def _container_reference(self) -> int:
        return self._variable.variablesReference

    def _children(self) -> list[VariableContext]:
        return self.children()

    def __str__(self) -> str:
        return f"variable '{self._variable.name}'"


@dataclass(frozen=True)
class CapturedOutput:
    seen_texts: str
    """The accumulated text until the terminator (included if it was an OutputEvent)."""
    event: Event
    """The event that terminated the collection"""


class _ConfigureContext:
    """Handles the initial launch sequence handshake.

    Orchestrates the full DAP initialization sequence:
    On enter:
        1. Request and respond to the `Initialize` command.
        2. Send launch/attach request.
        3. Wait for InitializedEvent.

    In between:
       The test can set breakpoints or perform any check it needs do.

    On exit:
        4. Set and verify the pending source and function breakpoints.
        5. Request and response to configurationDone.
        6. Wait for ProcessEvent and launch/attach response.

    Example:

      >>> session.configure(LaunchArgs(program="a.out")) as ctx:
      ...     session.resolve_function_breakpoints(["do_foo"])
      >>> session.wait_for_breakpoint(after=ctx.process_event)
    """

    def __init__(
        self,
        session: "DAPTestSession",
        config: LaunchArgs | AttachArgs,
    ):
        self._session = session
        self._config = config
        self._pending_request: Optional[PendingResponse[EmptyBodyResponse]] = None

    def __enter__(self) -> "_ConfigureContext":
        session = self._session
        session.test_case.assertFalse(
            session._state.is_initialized, "session already started."
        )
        self.init_response = session.initialize_sequence(session.initialize_args)
        self._pending_request = session.send_request(self._config)
        session.wait_for_event(InitializedEvent, after=self.init_response)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        session = self._session
        assert self.init_response is not None
        assert self._pending_request is not None

        session.verify_configuration_done()
        self.process_event = session.wait_for_event(
            ProcessEvent, after=self.init_response
        )

        self.launch_or_attach_response = self._pending_request.result()
        return False


@dataclass
class _ExpectCommon:
    """Shared fields used by both `ExpectVar` and `ExpectEval`.
    Any attribute set to `None` will be skipped when checking.
    """

    type: Optional[str] = None
    variables_reference: Optional[int] = None
    named_variables: Optional[int] = None
    indexed_variables: Optional[int] = None
    read_only: bool = False

    has_var_ref: Optional[bool] = None
    has_mem_ref: Optional[bool] = None
    has_loc_ref: Optional[bool] = None
    has_indexed_variables: Optional[bool] = None

    # Checks on the Expression's result or Variable's value.
    startswith: Optional[str] = None
    matches: Optional[str] = None  # regex applied to .value/.result
    # When set, fetch children via `variablesReference` and verify recursively.
    children: Optional[dict[str, "ExpectVar"]] = None


@dataclass
class ExpectVar(_ExpectCommon):
    """Typed expectation for a `Variable`.
    Any attribute set to `None` will be skipped when checking.
    """

    value: Optional[str] = None
    evaluate_name: Optional[str] = None
    has_evaluate_name: Optional[bool] = None


@dataclass
class ExpectEval(_ExpectCommon):
    """Typed expectation for an `EvaluateResponse.body`
    Any attribute set to `None` will be skipped when checking.
    """

    result: Optional[str] = None


class DAPTestSession(Session):
    """A `Session` bound to a `unittest.TestCase`.

    Adds repeating patterns for sending, receiving and verifying protocol messages.
    such as breakpoints, threads and evaluate.
    """

    def __init__(
        self,
        test_case: unittest.TestCase,
        test_dir: Path,
        adapter: DebugAdapter,
        message_timeout: float,
        process_spawner: SubProcessSpawner,
        logger: logging.Logger,
    ):
        super().__init__(test_dir, adapter, message_timeout, process_spawner, logger)
        self.test_case = test_case

        # The default features that lldb supports.
        # When a test does not explicitly set initialize args this is used.
        self._init_args = InitializeArgs(
            adapterID="lldb-native",
            clientID="vscode",
            columnsStartAt1=True,
            linesStartAt1=True,
            locale="en-us",
            pathFormat="path",
            supportsRunInTerminalRequest=True,
            supportsVariablePaging=True,
            supportsVariableType=True,
            supportsStartDebuggingRequest=True,
            supportsProgressReporting=True,
            supportsInvalidatedEvent=True,
            supportsMemoryEvent=True,
        )

    def update_initialize_args(self, **kwargs):
        self.test_case.assertFalse(
            self._state.is_initialized,
            "session already initialized cannot update initialize args.",
        )

        self._init_args = dataclasses.replace(self._init_args, **kwargs)

    @property
    def initialize_args(self):
        return dataclasses.replace(self._init_args)

    def launch(self, config: LaunchArgs) -> ProcessEvent:
        """Drives the full launch handshake

        (initialize -> launch -> configurationDone -> ProcessEvent).
        """
        with self.configure(config) as ctx:
            pass
        return ctx.process_event

    def attach(self, config: AttachArgs) -> ProcessEvent:
        """Drives the full attach handshake

        (initialize -> attach -> configurationDone -> ProcessEvent).
        """
        with self.configure(config) as ctx:
            pass
        return ctx.process_event

    def configure(self, config: LaunchArgs | AttachArgs) -> _ConfigureContext:
        """Return a context that scopes the launch sequence.

        process_event and launch_or_attach are only a valid after
        leaving the context block.

        Example:
            >>> with session.configure(LaunchArgs(program)) as ctx:
            ...     session.set_source_breakpoints("main.cpp", [10, 25])
            >>> process_event = ctx.process_event
            >>> response = ctx.launch_or_attach_response
        """
        return _ConfigureContext(self, config)

    def initialize_sequence(self, initialize_args: InitializeArgs):
        init_response = self.send_request(initialize_args).result()
        return init_response

    def initialize_and_launch(self, args: LaunchArgs | AttachArgs):
        self.initialize_sequence(self.initialize_args)
        return self.send_request(args)

    def configuration_done(self) -> PendingResponse[EmptyBodyResponse]:
        # Wait for initialized event.
        self.ensure_initialized()
        # And then send configuration done.
        return self.send_request(ConfigurationDoneArgs())

    def verify_configuration_done(self, expected_success: bool = True):
        response = self.configuration_done().result_or_error()
        if expected_success:
            self.test_case.assertEqual(
                response.success, True, f"got error response: {response}."
            )
            self.test_case.assertIsInstance(response, EmptyBodyResponse)

            # In VSCode, immediately following 'configurationDone', a
            # 'threads' request is made to get the initial set of threads,
            # specifically the main threads id and name.
            # We issue the threads request to mimic this pattern and prevent
            # tests that use threads to have the wrong result.
            self.send_request(ThreadsArgs()).result()
        else:
            self.test_case.assertEqual(response.success, False)
            self.test_case.assertIsInstance(response, ErrorResponse)
        return response

    def set_source_breakpoints(
        self, source_path: str, breakpoints: list[int] | list[SourceBreakpoint]
    ):
        self.ensure_initialized()
        # Convert the deprecated lines field to SourceBreakpoints.
        s_breakpoints: list[SourceBreakpoint] = []
        for bp in breakpoints:
            if isinstance(bp, int):
                s_breakpoints.append(SourceBreakpoint(bp))
            elif isinstance(bp, SourceBreakpoint):
                s_breakpoints.append(bp)
            else:
                self.test_case.fail(
                    "breakpoints must only contain ints or SourceBreakpoints."
                    f" got '{bp}' of type '{type(bp)}'."
                )

        source = Source.create(path=source_path)
        bp_args = SetBreakpointsArgs(source, breakpoints=s_breakpoints)
        return self.send_request(bp_args).result()

    def set_assembly_breakpoints(
        self,
        source: Source | int,
        breakpoints: list[int] | list[SourceBreakpoint],
    ):
        """Set breakpoints in an assembly source.

        `source` can be either a `sourceReference` int (for a source produced
        in the current session) or a full `Source` object (for replaying a
        persisted assembly source across sessions).
        """
        self.ensure_initialized()
        if isinstance(source, int):
            source = Source(sourceReference=source)

        s_breakpoints: list[SourceBreakpoint] = []
        for bp in breakpoints:
            if isinstance(bp, int):
                s_breakpoints.append(SourceBreakpoint(bp))
            elif isinstance(bp, SourceBreakpoint):
                s_breakpoints.append(bp)
            else:
                self.test_case.fail(
                    "breakpoints must only contain ints or SourceBreakpoints."
                    f" got '{bp}' of type '{type(bp)}'."
                )

        bp_args = SetBreakpointsArgs(source=source, breakpoints=s_breakpoints)
        return self.send_request(bp_args).result()

    def set_function_breakpoints(
        self, breakpoints: list[str] | list[FunctionBreakpoint]
    ):
        f_breakpoints: list[FunctionBreakpoint] = []
        for bp in breakpoints:
            if isinstance(bp, str):
                func_bp = FunctionBreakpoint(name=bp)
                f_breakpoints.append(func_bp)
            elif isinstance(bp, FunctionBreakpoint):
                f_breakpoints.append(bp)
            else:
                self.test_case.fail(
                    "breakpoints must only contain 'str' or 'FunctionBreakpoints'."
                    f" got '{bp}' of type '{type(bp)}'."
                )
        response = self.send_request(SetFunctionBreakpointsArgs(f_breakpoints)).result()
        return response

    def set_exception_breakpoints(
        self,
        *,
        filters: list[str],
        filterOptions: Optional[list[ExceptionFilterOptions]] = None,
        exceptionOptions: Optional[list[ExceptionOptions]] = None,
    ):
        args = SetExceptionBreakpointsArgs(
            filters=filters,
            filterOptions=filterOptions,
            exceptionOptions=exceptionOptions,
        )
        response = self.send_request(args).result()
        return response

    def set_variable(
        self, name: str, value, *, variablesReference: int, is_hex: bool = False
    ) -> SetVariableResponse | ErrorResponse:
        last_event = self.last_event()
        handle = self.send_request(
            SetVariableArgs(
                variablesReference=variablesReference,
                name=name,
                value=str(value),
                format=ValueFormat(hex=True) if is_hex else None,
            )
        )
        response = handle.result_or_error()
        if isinstance(response, SetVariableResponse):
            invalidated_event = self.wait_for_invalidated_event(after=last_event)
            invalidated_areas = invalidated_event.body.areas
            self.test_case.assertEqual(["variables"], invalidated_areas)

            memory_event = self.wait_for_memory_event(after=last_event)
            self.test_case.assertEqual(
                memory_event.body.memoryReference, response.body.memoryReference
            )
        return response

    @staticmethod
    def breakpoints_to_ids(breakpoints: list[Breakpoint]):
        ids: list[int] = []
        for bp in breakpoints:
            assert bp.id is not None, f"id is None for breakpoint: {bp}"
            ids.append(bp.id)
        return ids

    def resolve_source_breakpoints(
        self, source_path: str, breakpoints: list[int] | list[SourceBreakpoint]
    ) -> list[int]:
        last_event = self.last_event()
        bp_response = self.set_source_breakpoints(source_path, breakpoints)
        r_breakpoints = bp_response.body.breakpoints

        all_verified = all(bp.verified for bp in r_breakpoints)
        if not all_verified:
            self.wait_until_all_breakpoints_verified(r_breakpoints, after=last_event)

        self.test_case.assertEqual(
            len(breakpoints),
            len(r_breakpoints),
            "expect correct number of breakpoints.",
        )
        return self.breakpoints_to_ids(r_breakpoints)

    def resolve_function_breakpoints(
        self, breakpoints: list[str] | list[FunctionBreakpoint]
    ) -> list[int]:
        """Sets breakpoints by function name given an array of function names
        and returns an array of strings containing the breakpoint IDs
        ("1", "2") for each breakpoint that was set.
        """
        last_event = self.last_event()
        response = self.set_function_breakpoints(breakpoints)
        resp_bps = response.body.breakpoints

        all_verified = all(bp.verified for bp in resp_bps)
        if not all_verified:
            self.wait_until_all_breakpoints_verified(resp_bps, after=last_event)

        return self.breakpoints_to_ids(resp_bps)

    def set_data_breakpoints(self, breakpoints: list[DataBreakpoint]):
        args = SetDataBreakpointsArgs(breakpoints=breakpoints)
        return self.send_request(args).result()

    def set_instruction_breakpoints(self, memory_references: list[str]):
        breakpoints = [InstructionBreakpoint(ref) for ref in memory_references]
        return self.send_request(SetInstructionBreakpointsArgs(breakpoints)).result()

    def get_breakpoint_locations(
        self,
        file_path: str,
        line: int,
        column: Optional[int] = None,
        endLine: Optional[int] = None,
        endColumn: Optional[int] = None,
    ):
        _, name = os.path.split(file_path)
        bp_loc_args = BreakpointLocationsArgs(
            Source.create(name=name, path=file_path),
            line=line,
            column=column,
            endLine=endLine,
            endColumn=endColumn,
        )
        return self.send_request(bp_loc_args).result()

    def data_breakpoint_info(self, name: str, variablesReference: int, frameId: int):
        info_args = DataBreakpointInfoArgs(
            name=name, variablesReference=variablesReference, frameId=frameId
        )
        return self.send_request(info_args).result()

    def data_breakpoint_info_as_address(self, address: str, size: int):
        info_args = DataBreakpointInfoArgs(name=address, bytes=size, asAddress=True)
        return self.send_request(info_args).result()

    def step_in(
        self,
        threadId: int,
        *,
        targetId: Optional[int] = None,
        granularity: SteppingGranularity = "statement",
    ):
        stepin_args = StepInArgs(
            threadId=threadId, targetId=targetId, granularity=granularity
        )
        response = self.send_request(stepin_args).result()
        stop_event = self.verify_stopped(StoppedReason.STEP, after=response)
        return stop_event

    def step_over(
        self,
        threadId: int,
        *,
        granularity: SteppingGranularity = "statement",
    ):
        next_args = NextArgs(threadId=threadId, granularity=granularity)
        response = self.send_request(next_args).result()
        stop_event = self.verify_stopped(StoppedReason.STEP, after=response)
        return stop_event

    def step_out(
        self,
        threadId: int,
        *,
        granularity: SteppingGranularity = "statement",
    ):
        step_out_args = StepOutArgs(threadId=threadId, granularity=granularity)
        response = self.send_request(step_out_args).result()

        stop_event = self.verify_stopped(StoppedReason.STEP, after=response)
        return stop_event

    def wait_until_any_breakpoint_hit(
        self, breakpoint_ids: list[int], *, after: Event | Response
    ) -> StoppedEvent:
        """Wait for the process to send `StoppedEvents` and verify we stopped for
        any breakpoint in breakpoint_ids the event or response.
        """

        self.test_case.assertGreater(len(breakpoint_ids), 0, "empty breakpoint ids.")
        is_ids_int = all(isinstance(id, int) for id in breakpoint_ids)
        self.test_case.assertTrue(is_ids_int, "all breakpoint_ids must be integers.")

        breakpoint_stop_reasons = [
            StoppedReason.BREAKPOINT,
            StoppedReason.INSTRUCTION_BREAKPOINT,
            StoppedReason.FUNCTION_BREAKPOINT,
            StoppedReason.DATA_BREAKPOINT,
        ]

        def event_hit_id_in_breakpoint_ids(event: StoppedEvent):
            hit_ids = event.body.hitBreakpointIds or []
            for hit_id in hit_ids:
                if hit_id in breakpoint_ids:
                    return True

            return False

        event = self.wait_for_stopped_event(
            breakpoint_stop_reasons,
            after=after,
            until=event_hit_id_in_breakpoint_ids,
            timeout_msg=f"waiting for any breakpoint id in {breakpoint_ids} after seq {after.seq}.",
        )

        return event

    def wait_until_all_breakpoints_verified(
        self, breakpoints: list[int] | list[Breakpoint], *, after: Event | Response
    ):
        """Wait for the process to send breakpoint events and verify we hit
        all 'ids' in 'breakpoints' after the event or response.
        """
        self.test_case.assertTrue(len(breakpoints) > 0, "empty list of breakpoints.")

        id_to_verify: dict[int, bool] = {}
        for bp in breakpoints:
            if isinstance(bp, int):
                id_to_verify[bp] = False
            elif isinstance(bp, Breakpoint):
                assert bp.id is not None
                id_to_verify[bp.id] = bp.verified
            else:
                self.test_case.fail(
                    f"expected list of type 'Breakpoint' or 'int' got '{breakpoints}'"
                )

        bp_ids = list(id_to_verify.keys())

        def all_breakpoints_verified(evt: BreakpointEvent):
            event_bp = evt.body.breakpoint
            if event_bp.id is None:
                return False

            if event_bp.id not in bp_ids:
                return False

            id_to_verify[event_bp.id] = event_bp.verified
            all_verified = all(verified for verified in id_to_verify.values())
            return all_verified

        timeout_msg = f"waiting for all breakpoint ids {bp_ids} to be verified"
        last_breakpoint_event = self.wait_for_event(
            BreakpointEvent,
            after=after,
            until=all_breakpoints_verified,
            timeout_msg=timeout_msg,
        )
        return last_breakpoint_event

    def wait_for_stopped_or_exited_event(
        self,
        *,
        after: Event | Response,
        until: Optional[Callable[[StoppedEvent | ExitedEvent], bool]] = None,
        timeout_msg: Optional[str] = None,
    ) -> StoppedEvent | ExitedEvent:
        event = self.wait_for_any_event(
            (StoppedEvent, ExitedEvent),
            after=after,
            until=until,
            timeout_msg=timeout_msg,
        )
        return event

    def wait_for_stopped_event(
        self,
        matching_any: Optional[Sequence[StoppedReason]] = None,
        *,
        after: Event | Response,
        until: Optional[Callable[[StoppedEvent], bool]] = None,
        timeout_msg: Optional[str] = None,
    ) -> StoppedEvent:
        """
        Wait for a process to stop, optionally filtered by stop reason and custom condition.

        Blocks until a StoppedEvent is received after the specified event. If matching_any
        is provided, only stops with those reasons are accepted. The until callback allows
        additional condition checking. If an ExitedEvent is encountered, wait_for terminates.

        Args:
            matching_any: Filter by specific stop reasons.
            after: Event or Response to start waiting after.
            until: Optional callback for additional filtering.
            timeout_msg: Custom timeout error message.
        """
        if matching_any:
            self.test_case.assertGreater(
                len(matching_any), 0, "expected at least one stop reason."
            )

        def matches_any_reason_until(event: StoppedEvent | ExitedEvent):
            # Break early for exited event.
            # We cannot hit a stopped event after the process exited.
            if isinstance(event, ExitedEvent):
                return True

            # Match any of the stopped reasons.
            if matching_any and event.body.reason not in matching_any:
                return False

            if until:
                return until(event)

            return True

        event = self.wait_for_stopped_or_exited_event(
            after=after, until=matches_any_reason_until, timeout_msg=timeout_msg
        )

        self.test_case.assertIsInstance(event, StoppedEvent, f"after seq: {after.seq}")
        self.test_case.assertEqual(event.event, EventName.STOPPED)
        return cast(StoppedEvent, event)

    def wait_for_exited_event(self, *, after: Event | Response) -> ExitedEvent:
        """
        Wait for a process to exit.

        Blocks until an ExitedEvent is received following the given event or response.
        Raises an error if a StoppedEvent is encountered, as a stopped process
        cannot subsequently exit.
        """
        event = self.wait_for_stopped_or_exited_event(after=after)
        self.test_case.assertIsInstance(event, ExitedEvent)
        self.test_case.assertEqual(event.event, "exited", "expected ExitedEvent'")
        return cast(ExitedEvent, event)

    def verify_next_module_event(
        self,
        reason: Optional[ModuleReason] = None,
        *,
        after: Event | Response,
    ):
        event = self.wait_for_module_event(after=after)
        event_body = event.body
        if reason is not None:
            msg = f"module event reason does not match, got {event_body}."
            self.test_case.assertEqual(event_body.reason, reason, msg)
        return event

    def wait_for_module_event(
        self,
        *,
        after: Event | Response,
        until: Optional[Callable[[ModuleEvent], bool]] = None,
    ):
        return self.wait_for_event(ModuleEvent, after=after, until=until)

    def wait_for_terminated_event(self, *, after: Event | Response):
        return self.wait_for_event(TerminatedEvent, after=after)

    def wait_for_invalidated_event(self, *, after: Event | Response):
        return self.wait_for_event(InvalidatedEvent, after=after)

    def wait_for_memory_event(self, *, after: Event | Response):
        return self.wait_for_event(MemoryEvent, after=after)

    def do_continue(self):
        """Send a 'continueRequest' and wait for a 'ContinuedEvent',

        Receiving the continue response does not mean the process continued,
        It means we successfully sent the continue packet.
        LLDB can continue asynchronously, so wait for the 'Continued' event.
        """
        self.ensure_initialized()
        prior_event = self.last_event()
        response = self.send_request(ContinueArgs()).result()
        self.wait_for_event(ContinuedEvent, after=prior_event)
        return response

    def continue_to_exit(self, exitCode: int = 0) -> ExitedEvent:
        continue_response = self.do_continue()
        return self.verify_process_exited(after=continue_response, exitCode=exitCode)

    def continue_to_breakpoint(self, breakpoint_id: int):
        return self.continue_to_any_breakpoint([breakpoint_id])

    def continue_to_any_breakpoint(self, breakpoint_ids: list[int]):
        response = self.do_continue()
        event = self.wait_until_any_breakpoint_hit(breakpoint_ids, after=response)
        return event

    def continue_to_exception_breakpoint(
        self,
        *,
        expected_description: Optional[str] = None,
        expected_text: Optional[str] = None,
    ):
        continue_response = self.do_continue()
        return self.verify_stopped_on_exception(
            expected_description=expected_description,
            expected_text=expected_text,
            after=continue_response,
        )

    def continue_to_next_stop(self, *, exp_reason: Optional[StoppedReason] = None):
        """Continue execution and wait for stopped event"""
        response = self.do_continue()
        if exp_reason is None:
            return self.wait_for_stopped_event(after=response)

        return self.verify_stopped(exp_reason, after=response)

    def evaluate(
        self,
        expression: str,
        *,
        frameId: Optional[int] = None,
        context: Optional[EvaluateContext] = None,
        format: Optional[ValueFormat] = None,
    ):
        """Send an `evaluate` request and expects a successful response and result."""
        pending = self.do_evaluate(
            expression, frameId=frameId, context=context, format=format
        )
        response = pending.result(
            f"failed to evaluate `{expression}` with {context=}, {frameId=}."
        )

        result = response.body.result
        self.test_case.assertFalse(result.startswith("error:"), f'"{result}"')
        return response.body

    def do_evaluate(
        self,
        expression: str,
        *,
        frameId: Optional[int] = None,
        context: Optional[EvaluateContext] = None,
        format: Optional[ValueFormat] = None,
    ) -> PendingResponse[EvaluateResponse]:
        """Send an `evaluate` request without failing on error."""
        eval_args = EvaluateArgs(
            expression=expression, frameId=frameId, context=context, format=format
        )
        return self.send_request(eval_args)

    def collect_output(
        self,
        category: OutputCategory,
        *,
        until: str | Event,
        after: Event | Response,
        timeout_msg: Optional[str] = None,
    ) -> CapturedOutput:
        """Collect OutputEvents in `category` until a terminator is reached.

        Args:
            until:
                If a `str`, return once this substring is seen in an
                OutputEvent's text in `category`.
                If an `Event` instance, return when that exact event (matched by seq) is reached —
                useful after a prior `wait_for_event` call gave you the
                synchronization point. The terminator's own text is not
                accumulated.
            category: The output category to collect.
        Returns:
            CapturedOutput. `event` is the terminator — the matching
            OutputEvent (string form) or the supplied event (instance form).
        """
        if not isinstance(until, (str, Event)):
            self.test_case.fail(f"expected str or Event for until got {type(until)}")

        seen_outputs: list[str] = []

        if isinstance(until, str):
            self.test_case.assertTrue(until, "expected a non-empty pattern")
            pattern = until

            def matches_pattern(event: OutputEvent):
                if event.body.category != category:
                    return False
                seen_outputs.append(event.body.output)
                return pattern in event.body.output

            timeout_msg = f"{timeout_msg}\n\t" if timeout_msg else ""
            timeout_msg += f"collecting output category '{category}' until found pattern: '{pattern}'."

            event = self.wait_for_event(
                OutputEvent,
                after=after,
                until=matches_pattern,
                timeout_msg=timeout_msg,
            )
            # Sanity check.
            self.test_case.assertIsInstance(event, OutputEvent)
            return CapturedOutput(seen_texts="".join(seen_outputs), event=event)

        until_event = until
        self.test_case.assertGreater(
            until_event.seq,
            after.seq,
            f"'{until_event}' event must be later than '{after}'.",
        )

        def matches_until_event(event: Event):
            if isinstance(event, OutputEvent) and event.body.category == category:
                seen_outputs.append(event.body.output)
            return event.seq >= until_event.seq

        timeout_msg = f"{timeout_msg}\n\t" if timeout_msg else ""
        timeout_msg += f"collecting output category '{category}' up to event seq "
        timeout_msg += f"{until_event.seq} ({type(until_event).__name__})."

        # The until_event is already in history, so this never blocks beyond
        # reading the events that are already recorded.
        self.wait_for_event(
            Event,
            after=after,
            until=matches_until_event,
            timeout_msg=timeout_msg,
        )
        return CapturedOutput(seen_texts="".join(seen_outputs), event=until_event)

    def collect_console(self, *, until: str | Event, after: Event | Response):
        return self.collect_output(OutputCategory.CONSOLE, until=until, after=after)

    def collect_stdout(self, *, until: str | Event, after: Event | Response):
        return self.collect_output(OutputCategory.STDOUT, until=until, after=after)

    def collect_important(self, *, until: str | Event, after: Event | Response):
        return self.collect_output(OutputCategory.IMPORTANT, until=until, after=after)

    def verify_stopped(
        self,
        reasons: StoppedReason | Iterable[StoppedReason],
        *,
        after: Event | Response,
        expected_ids: Iterable[int] | int | None = None,
        expected_description: Optional[str] = None,
        expected_text: Optional[str] = None,
    ) -> StoppedEvent:
        """Wait for a `StoppedEvent` and assert the body matches every supplied
        expectation."""
        if isinstance(reasons, StoppedReason):
            reasons = [reasons]

        timeout_msg = f"waiting for 'StoppedEvent' matching reasons: {reasons}"
        stopped_event = self.wait_for_stopped_event(
            after=after, timeout_msg=timeout_msg
        )

        body = stopped_event.body
        test_case = self.test_case
        msg = f"for stopped event {body}."
        test_case.assertIn(body.reason, reasons, msg)

        if expected_ids is not None:
            if isinstance(expected_ids, int):
                expected_ids = [expected_ids]

            hit_bp_ids = body.hitBreakpointIds or []
            for expected_id in expected_ids:
                msg = f"expected breakpoint_id '{expected_id}' not in {hit_bp_ids=}."
                test_case.assertIn(expected_id, hit_bp_ids, msg)

        if expected_description is not None:
            test_case.assertIsNotNone(
                body.description, f"stopped event missing description {body}."
            )
            description = cast(str, body.description)
            test_case.assertRegex(description, expected_description, msg)

        if expected_text is not None:
            test_case.assertIsNotNone(body.text, f"stopped event missing text {body}.")
            text = cast(str, body.text)
            test_case.assertRegex(text, expected_text, msg)

        return stopped_event

    def verify_stopped_on_breakpoint(
        self,
        expected_ids: list[int] | int | None = None,
        *,
        after: Event | Response,
    ) -> StoppedEvent:
        reasons = [
            StoppedReason.BREAKPOINT,
            StoppedReason.DATA_BREAKPOINT,
            StoppedReason.FUNCTION_BREAKPOINT,
            StoppedReason.INSTRUCTION_BREAKPOINT,
        ]
        return self.verify_stopped(reasons, after=after, expected_ids=expected_ids)

    def verify_stopped_on_entry(self, *, after: Event | Response) -> StoppedEvent:
        return self.verify_stopped(StoppedReason.ENTRY, after=after)

    def verify_stopped_on_exception(
        self,
        *,
        expected_description: Optional[str] = None,
        expected_text: Optional[str] = None,
        after: Event | Response,
    ) -> StoppedEvent:
        """Wait for the debuggee to stop with reason `exception` and verify
        the description matches `expected_description` (regex) and, if given,
        the text matches `expected_text` (regex).
        """
        return self.verify_stopped(
            StoppedReason.EXCEPTION,
            after=after,
            expected_description=expected_description,
            expected_text=expected_text,
        )

    def verify_multiple_breakpoints_hit(
        self, breakpoint_ids: list[int], *, after: Event | Response
    ) -> StoppedEvent:
        """Wait for the session receive a 'StoppedEvent' and verify we stopped for
        any breakpoint in breakpoint_ids the event or response.
        """

        self.test_case.assertGreater(len(breakpoint_ids), 0, "empty breakpoint ids.")
        is_ids_int = all(isinstance(id, int) for id in breakpoint_ids)
        self.test_case.assertTrue(is_ids_int, "all breakpoint_ids must be integers.")

        event = self.verify_stopped_on_breakpoint(after=after)
        hit_ids = event.body.hitBreakpointIds or []
        if set(breakpoint_ids).issubset(hit_ids):
            return event

        self.test_case.fail(f"{breakpoint_ids=} missed in {event=} {after=}.")

    def verify_process_exited(
        self, *, after: Event | Response | None = None, exitCode: int = 0
    ):
        if after:
            event = self.wait_for_exited_event(after=after)
        else:
            event = self.wait_for_earliest_event(ExitedEvent)

        fail_msg = f"expect '{exitCode=}' for '{event.body}'"
        self.test_case.assertEqual(event.body.exitCode, exitCode, fail_msg)

        self.verify_reverse_process_exited(exitCode)
        return event

    def verify_commands(self, flavor: str, output: str, commands: list[str]):
        self.test_case.assertTrue(output and len(output) > 0, "expect console output")
        lines = output.splitlines()
        prefix = "(lldb) "

        for cmd in commands:
            cmd_stripped = cmd.lstrip("!?")
            for line in lines:
                if line.startswith(prefix) and cmd_stripped in line:
                    break
            else:
                self.test_case.fail(
                    f"Command '{flavor}' - '{cmd}' not found in output: {output}."
                )

    def verify_location(self, locationReference: int, filename: str, line: int):
        response = self.send_request(LocationsArgs(locationReference)).result()
        path = response.body.source.path
        self.test_case.assertIsNotNone(response.body.source.path)

        msg = f"expect path '{path}' to end with '{filename}'."
        self.test_case.assertTrue(str(path).endswith(filename), msg)
        self.test_case.assertEqual(response.body.line, line)

    def __verify_common(
        self, actual: Variable | EvaluateResponse.Body, expected: _ExpectCommon
    ):
        """Verify the shared fields of `Variable` and `EvaluateResponse.Body`."""
        if expected.type is not None:
            msg = f"type mismatch for {actual}: {actual.type!r} != {expected.type!r}."
            self.test_case.assertEqual(actual.type, expected.type, msg)

        if expected.variables_reference is not None:
            self.test_case.assertEqual(
                actual.variablesReference,
                expected.variables_reference,
                f"variablesReference mismatch for {actual}.",
            )

        if expected.named_variables is not None:
            self.test_case.assertEqual(
                actual.namedVariables,
                expected.named_variables,
                f"namedVariables mismatch for {actual}.",
            )

        if expected.indexed_variables is not None:
            self.test_case.assertEqual(
                actual.indexedVariables,
                expected.indexed_variables,
                f"indexedVariables mismatch for {actual}.",
            )

        if expected.has_var_ref is not None:
            has_var_ref = bool(actual.variablesReference)
            fail_msg = f"has_var_ref mismatch for {actual=}, {expected=}."
            self.test_case.assertEqual(has_var_ref, expected.has_var_ref, fail_msg)

        if expected.has_mem_ref is not None:
            has_mem_ref = actual.memoryReference is not None
            fail_msg = f"has_mem_ref mismatch for {actual=}, {expected=}."
            self.test_case.assertEqual(has_mem_ref, expected.has_mem_ref, fail_msg)

        if expected.has_loc_ref is not None:
            has_loc_ref = actual.valueLocationReference is not None
            fail_msg = f"has_loc_ref mismatch for {actual=}, {expected=}."
            self.test_case.assertEqual(has_loc_ref, expected.has_loc_ref, fail_msg)

        if expected.has_indexed_variables is not None:
            has_idx_vars = actual.indexedVariables is not None
            fail_msg = f"has_index_variables mismatch for {actual=}, {expected=}."
            self.test_case.assertEqual(
                has_idx_vars, expected.has_indexed_variables, fail_msg
            )

        hint = actual.presentationHint or VariablePresentationHint()
        attributes = hint.attributes or []
        fail_msg = f"readOnly attribute mismatch for {actual=}."
        if expected.read_only:
            self.test_case.assertIn("readOnly", attributes, fail_msg)
        else:
            self.test_case.assertNotIn("readOnly", attributes, fail_msg)

        if expected.children is not None:
            var_ref = actual.variablesReference
            self.test_case.assertTrue(
                var_ref, f"children expected but no variablesReference for {actual=}."
            )
            children = self.get_variables(var_ref)
            self.verify_variables(children, expected.children)

    def verify_evaluate(
        self,
        eval_body: EvaluateResponse.Body,
        expected: Optional[ExpectEval] = None,
        /,
        **expected_kwargs,
    ):
        """Verify an `EvaluateResponse.body`."""
        if expected is not None and expected_kwargs:
            self.test_case.fail("pass an ExpectEval OR its keyword fields, not both.")
        if expected is None and not expected_kwargs:
            self.test_case.fail("pass an ExpectEval or at least one keyword field.")

        expected = expected or ExpectEval(**expected_kwargs)
        eval_result = eval_body.result
        if expected.result is not None:
            self.test_case.assertEqual(
                eval_result, expected.result, f"result mismatch for {eval_body}."
            )

        if expected.matches is not None:
            self.test_case.assertRegex(eval_result, expected.matches)

        if expected_prefix := expected.startswith:
            self.test_case.assertTrue(
                eval_result.startswith(expected_prefix),
                f"'{eval_result!r}' does not start with '{expected_prefix!r}'.",
            )

        self.__verify_common(eval_body, expected)

    def verify_variable(
        self,
        variable: Variable,
        expected: Optional[ExpectVar] = None,
        /,
        **expected_kwargs,
    ):
        """Verify a Variable matches the expected `ExpectVar` ."""
        if expected is not None and expected_kwargs:
            self.test_case.fail("pass an ExpectVar or its keyword fields, not both.")
        if expected is None and not expected_kwargs:
            self.test_case.fail("pass an ExpectVar or at least one keyword field.")

        expected = expected or ExpectVar(**expected_kwargs)
        value = variable.value

        if expected.value is not None:
            fail_msg = f"value mismatch for {variable=}."
            self.test_case.assertEqual(value, expected.value, fail_msg)

        if expected.matches is not None:
            fail_msg = f"value doesn't match pattern for '{variable}'."
            self.test_case.assertRegex(variable.value, expected.matches, fail_msg)

        if expected_prefix := expected.startswith:
            fail_msg = f"{value!r} does not start with {expected_prefix!r}."
            self.test_case.assertTrue(value.startswith(expected_prefix), fail_msg)

        if evaluate_name := expected.evaluate_name:
            fail_msg = f"evaluateName mismatch for {variable}."
            self.test_case.assertEqual(variable.evaluateName, evaluate_name, fail_msg)

        if expected.has_evaluate_name is not None:
            has_evaluate_name = variable.evaluateName is not None
            fail_msg = f"has_evaluate_name mismatch for {variable=}, {expected=}."
            self.test_case.assertEqual(
                has_evaluate_name, expected.has_evaluate_name, fail_msg
            )

        self.__verify_common(variable, expected)

    def verify_variables(
        self, variables: list[Variable], expected: dict[str, ExpectVar]
    ):
        """Verify each `Variable` in `variables` against its entry in `expected`."""
        self.test_case.assertTrue(len(variables) >= 1, f"no variables to verify.")

        for variable in variables:
            if variable.name.startswith("std::"):
                continue
            self.test_case.assertIn(variable.name, expected)
            self.verify_variable(variable, expected[variable.name])

    def get_modules(
        self, startModule: Optional[int] = None, moduleCount: Optional[int] = None
    ):
        args = ModulesArgs(startModule=startModule, moduleCount=moduleCount)
        response = self.send_request(args).result()
        modules_dict = {module.name: module for module in response.body.modules}
        return modules_dict

    def get_threads(self) -> list[ThreadContext]:
        response = self.send_request(ThreadsArgs()).result()
        threads = response.body.threads
        t_threads = [ThreadContext(thread.id, self) for thread in threads]
        return t_threads

    def get_variables(
        self,
        variablesReference: int,
        *,
        filter: Optional[Literal["indexed", "named"]] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
        format: Optional[ValueFormat] = None,
    ) -> list[Variable]:
        args = VariablesArgs(
            variablesReference=variablesReference,
            filter=filter,
            start=start,
            count=count,
            format=format,
        )
        response = self.send_request(args).result(
            f"failed to get variables for reference: {variablesReference}"
        )
        return response.body.variables

    def thread_context_from(
        self, thread_ref: int | StoppedEvent | InvalidatedEvent
    ) -> ThreadContext:
        if isinstance(thread_ref, (StoppedEvent, InvalidatedEvent)):
            self.test_case.assertIsNotNone(thread_ref.body.threadId)
            thread_id = cast(int, thread_ref.body.threadId)
        elif isinstance(thread_ref, int):
            thread_id = thread_ref
        else:
            ref_type_name = type(thread_ref).__name__
            self.test_case.fail(f"cannot get thread context from '{ref_type_name}'.")
        return ThreadContext(thread_id, self)

    def top_frame_from(
        self, thread_ref: int | StoppedEvent | InvalidatedEvent
    ) -> FrameContext:
        """Top FrameContext of the currently stopped thread."""
        return self.thread_context_from(thread_ref).top_frame()

    def get_completions(self, text: str, frameId: Optional[int]):
        def code_units(input: str) -> int:
            utf16_bytes = input.encode("utf-16-le")
            # one UTF16 codeunit = 2 bytes.
            return len(utf16_bytes) // 2

        com_args = CompletionsArgs(
            text=text, column=code_units(text) + 1, frameId=frameId
        )
        response = self.send_request(com_args).result()
        return response.body.targets

    def get_exception_info(self, threadId: int):
        info_args = ExceptionInfoArgs(threadId=threadId)
        response = self.send_request(info_args).result()
        return response.body

    def do_restart(self, arguments: LaunchArgs | AttachArgs | None = None):
        restart_args = RestartArgs(arguments)
        return self.send_request(restart_args).result()

    def disassemble(
        self,
        memoryReference: str,
        instructionOffset: int = -50,
        instructionCount: int = 200,
        resolveSymbols: bool = True,
    ):
        dis_args = DisassembleArgs(
            memoryReference=memoryReference,
            instructionOffset=instructionOffset,
            instructionCount=instructionCount,
            resolveSymbols=resolveSymbols,
        )
        return self.send_request(dis_args).result().body.instructions

    def stack_trace(
        self,
        threadId: int,
        *,
        startFrame: Optional[int] = None,
        levels: Optional[int] = None,
        format: Optional[StackFrameFormat] = None,
    ):
        """Send a `stackTrace` request and wait for a response"""
        args = StackTraceArgs(
            threadId=threadId, startFrame=startFrame, levels=levels, format=format
        )
        return self.send_request(args).result()

    def read_memory(
        self, memoryReference: str, count: int, offset: Optional[int] = None
    ) -> PendingResponse[ReadMemoryResponse]:
        args = ReadMemoryArgs(
            memoryReference=memoryReference, offset=offset, count=count
        )
        return self.send_request(args)

    def write_memory(
        self,
        memoryReference: str,
        value: int | str | bytes,
        *,
        offset: Optional[int] = None,
        allowPartial: bool = False,
    ):
        """Send a `writeMemory` request.

        Integer value is serialized as little-endian bytes,
        `value` is Base64-encoded because the DAP protocol requires it.
        """
        if isinstance(value, int):
            # The minimum bytes needed to represent 'value'.
            byte_length = max(1, (value.bit_length() + 7) // 8)
            is_negative = value < 0
            val_bytes = value.to_bytes(byte_length, "little", signed=is_negative)
        elif isinstance(value, str):
            val_bytes = value.encode()
        else:
            val_bytes = value
        data = base64.b64encode(val_bytes).decode()

        before_request = self.last_event()
        write_args = WriteMemoryArgs(
            memoryReference=memoryReference,
            data=data,
            offset=offset,
            allowPartial=allowPartial,
        )
        handle = self.send_request(write_args)
        response = handle.result_or_error()

        # Check we sent invalidated event.
        if response.success and self.initialize_args.supportsInvalidatedEvent:
            invalidated = self.wait_for_invalidated_event(after=before_request)
            self.test_case.assertEqual(invalidated.body.areas, ["all"])
        return response

    def disconnect(
        self, restart: Optional[bool] = None, terminateDebuggee: Optional[bool] = None
    ):
        args = DisconnectArgs(restart=restart, terminateDebuggee=terminateDebuggee)
        response = self.send_request(args).result()
        return response
