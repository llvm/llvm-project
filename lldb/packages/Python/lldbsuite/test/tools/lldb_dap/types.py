# NOTE: this module must not include `from __future__ import annotations`
# as the `annotations` import changes some of the types hints to strings.
# especially when you have a forward declared type reference.
# see https://peps.python.org/pep-0649/#motivation-for-this-pep
# https://peps.python.org/pep-0749/#rejected-alternatives
#
# This module may not depend on any other module.

from contextlib import suppress
import copy
import dataclasses
import enum
import json
import os
import sys
import typing
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""

        def __str__(self) -> str:
            return self.value

        def __repr__(self) -> str:
            return self.value

        @staticmethod
        def _generate_next_value_(name: str, start, count, last_values) -> str:
            return name.lower()


T = TypeVar("T")


class DAPError(AssertionError):
    """The base error for all DAP related errors

    Inherits from assertion error because of unittests treats assertions outside of a test as a failure
    instead of a test error. see https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUp
    """

    @classmethod
    def history_closed(cls, reason=None, last_event: Optional["Event"] = None):
        suffix = f" (Reason: {reason})" if reason else ""
        return cls(
            f"EventHistory is closed{suffix}. Last recorded event: {last_event}."
        )


RawMessage = Dict[str, Any]
"""Representation of a json protocol message """


class MessageType(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"


class EventName(StrEnum):
    BREAKPOINT = "breakpoint"
    CAPABILITIES = "capabilities"
    CONTINUED = "continued"
    EXITED = "exited"
    INITIALIZED = "initialized"
    INVALIDATED = "invalidated"
    MEMORY = "memory"
    MODULE = "module"
    OUTPUT = "output"
    PROCESS = "process"
    PROGRESS_END = "progressEnd"
    PROGRESS_START = "progressStart"
    PROGRESS_UPDATE = "progressUpdate"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    THREAD = "thread"


@dataclass(frozen=True)
class ProtocolMessage:
    type: MessageType
    seq: int

    def to_dict(self):
        return message_to_dict(self)

    @classmethod
    def from_json(cls: Type[T], json: RawMessage) -> T:
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f"{cls.__name__} must be a dataclass")

        return dict_to_message(cls, json)


@dataclass(frozen=True)
class Request(ProtocolMessage):
    command: str
    arguments: Any

    def __post_init__(self):
        assert (
            self.type == MessageType.REQUEST
        ), f"expected request type to be 'request' got '{self.type}' in : {self}"


@dataclass(frozen=True)
class Response(ProtocolMessage):
    command: str
    request_seq: int
    success: bool

    def __post_init__(self):
        assert (
            self.type == MessageType.RESPONSE
        ), f"expected '{type(self).__name__}' to be of type response: {self}."

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__!r} must be a dataclass.")


AnyResponse = TypeVar("AnyResponse", bound=Response)


class RequestError(DAPError):
    """Raised if a DAP request fails."""

    def __init__(
        self,
        request: Union[Request, dict],
        response: Optional[Union[Response, dict]] = None,
    ):
        super().__init__()
        self.request = request
        self.response = response

    def __str__(self) -> str:
        desc = f"request failed request={self.request!r}"
        if self.response:
            desc += f" response={self.response!r}"
        return desc


@dataclass(frozen=True)
class Event(ProtocolMessage):
    event: Union[EventName, str]
    type: MessageType
    __registry: ClassVar[Dict[str, Type]] = {}

    def __init_subclass__(cls, *, event: str, **kwargs):
        super().__init_subclass__(**kwargs)

        assert event is not None
        # Attach metadata (not a field of an event).
        cls.__message_type__ = event

        # Prevent duplicate types.
        existing_event_class = Event.__registry.get(event)
        if existing_event_class is not None:
            raise Exception(
                f"cannot register '{event}' event to class '{cls}' because it is already registered to '{existing_event_class}'"
            )

        Event.__registry[event] = cls

    def __post_init__(self):
        if self.type != MessageType.EVENT:
            raise ValueError(f"event must have type of 'event': {self}.")

    @classmethod
    def from_json(cls: Type[T], json: dict) -> T:
        event_type = json["event"]

        if event_type not in Event.__registry:
            raise ValueError(f"event type '{event_type}' is not registered.")

        event_class = Event.__registry[event_type]

        if cls not in (Event, event_class):
            raise ValueError(
                f"class: {cls.__name__!r} is not of the expected type {event_class.__name__!r}."
            )

        if not dataclasses.is_dataclass(event_class):
            raise ValueError(f"{cls.__name__!r} must be a dataclass.")

        return cast(T, dict_to_message(event_class, json))


AnyEvent = TypeVar("AnyEvent", bound=Event)


@dataclass(frozen=True)
class EmptyBodyResponse(Response):
    body: None = field(init=False, default=None)


def args_protocol(cls):
    """Decorator to check a class conforms to the ArgsProtocol"""

    required_fields = filter(lambda x: not x.startswith("_"), dir(ArgsProtocol))
    for r_field in required_fields:
        if not hasattr(cls, r_field):
            raise AttributeError(
                f"{cls.__name__} must define '{r_field}' to implement ArgProtocol"
            )

    command_name = getattr(cls, "command_")
    if not issubclass(type(command_name), str):
        raise TypeError(
            f"the command_ type '{type(command_name)}' for class '{cls.__name__}' must be string like"
        )

    return cls


def _message_to_dict_impl(obj: typing.Any, skip_none: bool = True) -> typing.Any:
    if dataclasses.is_dataclass(obj):
        fields = _get_dataclass_fields(type(obj))
        visited: Dict[str, Any] = {}
        for f in fields:
            dict_name = f.metadata.get("alias", f.name)
            if dict_name in visited or not hasattr(obj, f.name):
                continue
            name_attr = getattr(obj, f.name)
            if skip_none and name_attr is None:
                continue
            visited[dict_name] = _message_to_dict_impl(name_attr, skip_none)
        return visited

    if isinstance(obj, (list, tuple)):
        return type(obj)(_message_to_dict_impl(item, skip_none) for item in obj)

    if isinstance(obj, dict):
        result: Dict[str, Any] = {}
        for key, value in obj.items():
            result[str(key)] = _message_to_dict_impl(value, skip_none)
        return result

    # Test enum first as it can also be a subclass of other primitives.
    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, (bool, int, float, str, bytes, type, type(None))):
        return obj

    return copy.deepcopy(obj)


def message_to_dict(args: Any) -> RawMessage:
    """
    Converts DAP types to dictionaries.
    We always skip optional types in dataclasses during conversion.
    """
    if not is_dataclass(args):
        raise TypeError(
            f"expected a dataclass instance, got {type(args).__name__}: {args!r}"
        )
    result = _message_to_dict_impl(args)
    assert isinstance(result, dict)
    return result


@lru_cache
def _get_dataclass_fields(cls: Type) -> Tuple[dataclasses.Field, ...]:
    data_class_hints = typing.get_type_hints(cls)
    result: List[dataclasses.Field] = []
    for f in dataclasses.fields(cls):  # noqa
        # Ignores 'command_' and 'response_class_'.
        if f.name.endswith("_"):
            continue
        f_copy = copy.copy(f)
        f_copy.type = data_class_hints[f_copy.name]
        result.append(f_copy)
    return tuple(result)


@lru_cache
def _prepare_dataclass_fields(cls: Type) -> Dict[str, dataclasses.Field]:
    # Excludes init=False fields since they cannot be passed to __init__.
    return {f.name: f for f in _get_dataclass_fields(cls) if f.init}


def _get_compatible_union_types(data: Any, possible_types: List[Type]) -> List[Type]:
    """Filters a list of candidate types to find those compatible with the given raw data.

    This helps fail early on type mismatches during parsing. For example, it
    ensures that a raw payload like `[10, 20]` is correctly matched to `List[int]`
    rather than a primitive `int`.
    """

    def _is_type_compatible(candidate_type: Type, data: Any) -> bool:
        """Helper function to check if a single type is compatible with the data."""
        origin = typing.get_origin(candidate_type)

        if dataclasses.is_dataclass(candidate_type) or origin is dict:
            return isinstance(data, dict)

        if origin in (list, tuple):
            return isinstance(data, (list, tuple))

        if candidate_type is bool:
            return isinstance(data, bool)

        if candidate_type in (int, float):
            # Prevent implicit bool conversion as bool is a subclass of int.
            return isinstance(data, (int, float)) and not isinstance(data, bool)

        if candidate_type is str:
            return isinstance(data, str)

        if isinstance(candidate_type, type) and issubclass(candidate_type, enum.Enum):
            if issubclass(candidate_type, enum.IntEnum):
                return isinstance(data, int) and not isinstance(data, bool)
            return isinstance(data, str)

        # Fallback for unknown or other generic types
        return True

    result = [a_type for a_type in possible_types if _is_type_compatible(a_type, data)]
    return result


def _generic_to_message(cls: Optional[Type], data: Any, scope: List[str]) -> Any:
    origin = typing.get_origin(cls)
    args = typing.get_args(cls)
    full_path = ".".join(scope)

    if origin is Literal:
        unique_literals: Set = set()

        def flatten_literal(args: Any):
            """'Literal' types can be nested e.g.
            Literal[Literal["book"], Literal["pen"]] is the same as  Literal["book", "pen"]
            """
            for val in args:
                if typing.get_origin(val) is None:
                    unique_literals.add(val)
                else:
                    flatten_literal(typing.get_args(val))

        flatten_literal(args)

        if data not in unique_literals:
            raise ValueError(
                f"expected one of {unique_literals!r} at {full_path}, got {data!r}"
            )
        return data

    elif origin is Union:
        none_type = type(None)
        is_optional = none_type in args
        candidate_args = [a for a in args if a is not none_type]

        # Handle Optional.
        # Optional is represented as Union[T, None].
        if is_optional and len(candidate_args) == 1:
            return _dict_to_message_impl(candidate_args[0], data, scope)

        matches = _get_compatible_union_types(data, candidate_args)
        if not matches:
            raise TypeError(
                f"no variant of {cls} is compatible with "
                f"{type(data).__name__} at {full_path}: {data!r}"
            )

        # Only one match, try conversion.
        if len(matches) == 1:
            return _dict_to_message_impl(matches[0], data, scope)

        for arg in matches:
            with suppress(TypeError, ValueError, AttributeError, AssertionError):
                return _dict_to_message_impl(arg, data, scope)

        raise TypeError(
            f"no variant of {cls} matched {type(data).__name__} at {full_path}: {data!r}"
        )

    elif origin is list:
        if not isinstance(data, list):
            raise TypeError(
                f"expected list at {full_path}, got {type(data).__name__}: {data!r}"
            )
        list_type = args[0]
        return [
            _dict_to_message_impl(list_type, val, scope + [str(idx)])
            for idx, val in enumerate(data)
        ]

    elif origin is tuple:
        if not isinstance(data, list):
            raise TypeError(
                f"expected list for tuple at {full_path}, got {type(data).__name__}: {data!r}"
            )
        item_type = args[0]
        return tuple(
            _dict_to_message_impl(item_type, val, scope + [str(idx)])
            for idx, val in enumerate(data)
        )

    elif origin is dict:
        if not isinstance(data, dict):
            raise TypeError(
                f"expected dict at {full_path}, got {type(data).__name__}: {data!r}"
            )
        if len(args) == 0:
            args = (str, Any)
        key_type = args[0]
        value_type = args[1]
        return {
            _dict_to_message_impl(key_type, key, scope): _dict_to_message_impl(
                value_type, value, scope + [str(key)]
            )
            for key, value in data.items()
        }

    raise TypeError(f"unhandled generic type {cls} at {full_path}: {data!r}")


def _dict_to_message_impl(
    cls: Optional[Type], data: typing.Any, scope: List[str]
) -> typing.Any:
    """
    Recursively deserializes a dictionary into a specified Python type or dataclass.

    Args:
        cls: The target Python type or dataclass to deserialize the data into.
        data: The raw data (usually from a dictionary or JSON payload) to be converted.
        scope: A list of keys representing the current depth in the nested data structure.
               Used to provide error messages when validation fails.

    Returns:
        The deserialized data cast to the requested type or instantiated dataclass.

    Raises:
        TypeError: If the data type does not match the expected type and cannot be coerced.
        ValueError: If a dataclass field has a required value that does not match the data.
    """
    if not cls:
        return data

    full_path = ".".join(scope)
    if data is None:
        # Only pass None through if the declared type permits it.
        if cls is type(None):
            return None
        is_union = typing.get_origin(cls) is Union
        if is_union and type(None) in typing.get_args(cls):  # Is Optional
            return None

        raise TypeError(f"got None for non-optional type '{cls}' at {full_path}")

    if cls in (bytes, bytearray, Any):
        return data

    if dataclasses.is_dataclass(cls):
        if not isinstance(data, dict):
            raise TypeError(
                f"expected dict for '{cls.__name__}' at {full_path}, "
                f"got {type(data).__name__}: {data!r}"
            )
        fields = _prepare_dataclass_fields(cls)
        deserialized = {}
        for key, f in fields.items():
            data_key = f.metadata.get("alias", key)
            if data_key not in data:
                if (
                    f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING
                ):
                    raise TypeError(
                        f"expected field {data_key!r} in {data!r} at {full_path!r} for {cls.__name__!r}."
                    )
                continue

            required_value = f.metadata.get("required", dataclasses.MISSING)
            value = _dict_to_message_impl(f.type, data[data_key], scope + [key])

            if required_value is not dataclasses.MISSING and value != required_value:
                raise ValueError(
                    f"field '{key}' at {full_path} must be {required_value!r}, "
                    f"got {value!r}"
                )
            deserialized[key] = value
        try:
            return cls(**deserialized)
        except TypeError as err:
            msg = f"\n\tfailed to initialize '{cls.__name__}' at '{full_path}' "
            msg += f'\n\twith dict: "{data}"'
            err.args = (err.args[0] + msg, *err.args[1:])
            raise

    if cls in (str, int, float):
        if isinstance(data, cls):
            return data
        # Allow int to float coercion but not bool as bool is a subclass of int:
        if cls is float and isinstance(data, int) and not isinstance(data, bool):
            return float(data)
        raise TypeError(
            f"expected {cls.__name__} at {full_path}, "
            f"got {type(data).__name__}: {data!r}"
        )

    if cls is bool:
        if not isinstance(data, bool):
            raise TypeError(
                f"expected bool at {full_path}, " f"got {type(data).__name__}: {data!r}"
            )
        return data

    if isinstance(cls, type) and issubclass(cls, enum.Enum):
        if issubclass(cls, enum.IntEnum):
            return cls(int(data))
        return cls(data)

    # Handle generic types i.e List[T], Tuple[T] e.t.c.
    if typing.get_origin(cls) is not None:
        return _generic_to_message(cls, data, scope)

    raise TypeError(
        f"unexpected type {cls} at {full_path}: " f"got {type(data).__name__}: {data!r}"
    )


def dict_to_message(cls: Type[T], raw_dict: RawMessage) -> T:
    return _dict_to_message_impl(cls, raw_dict, [cls.__name__])


@runtime_checkable
class ArgsProtocol(Protocol[T]):  # type: ignore[misc]
    __dataclass_fields__: ClassVar[Dict]

    @property
    def response_class_(self) -> Type[T]:
        ...

    @property
    def command_(self) -> str:
        ...


class Console(StrEnum):
    INTERNAL = "internalConsole"
    INTEGRATED_TERMINAL = "integratedTerminal"
    EXTERNAL_TERMINAL = "externalTerminal"


class OutputCategory(StrEnum):
    CONSOLE = "console"
    IMPORTANT = "important"
    STDOUT = "stdout"
    STDERR = "stderr"
    TELEMETRY = "telemetry"


class StoppedReason(StrEnum):
    STEP = "step"
    BREAKPOINT = "breakpoint"
    EXCEPTION = "exception"
    PAUSE = "pause"
    ENTRY = "entry"
    GOTO = "goto"
    FUNCTION_BREAKPOINT = "function breakpoint"
    DATA_BREAKPOINT = "data breakpoint"
    INSTRUCTION_BREAKPOINT = "instruction breakpoint"


class ModuleReason(StrEnum):
    NEW = "new"
    CHANGED = "changed"
    REMOVED = "removed"


class BreakpointReason(StrEnum):
    NEW = "new"
    CHANGED = "changed"
    REMOVED = "removed"


class StartDebuggingRequestType(StrEnum):
    LAUNCH = "launch"
    ATTACH = "attach"


InvalidatedAreas = Literal["all", "stacks", "threads", "variables"]
ExceptionBreakMode = Literal["never", "always", "unhandled", "userUnhandled"]
ChecksumAlgorithm = Literal["MD5", "SHA1", "SHA256", "timestamp"]

ScopePresentationHint = Literal["arguments", "locals", "registers"]
SourcePresentationHint = Literal["normal", "emphasize", "deemphasize"]
StepInTargetPresentationHint = Literal["normal", "label", "subtle"]
GotoTargetPresentationHint = StepInTargetPresentationHint
StackFramePresentationHint = StepInTargetPresentationHint

VariablePresentationHintKind = Literal[
    "property",
    "method",
    "class",
    "data",
    "event",
    "baseClass",
    "innerClass",
    "interface",
    "mostDerivedClass",
    "virtual",
    "dataBreakpoint",
]

VariablePresentationHintVisibility = Literal[
    "public", "private", "protected", "internal", "final"
]


@dataclass(frozen=True)
class Checksum:
    algorithm: ChecksumAlgorithm
    checksum: str


@dataclass(frozen=True)
class Source:
    name: Optional[str] = None
    path: Optional[str] = None
    sourceReference: Optional[int] = None
    presentationHint: Optional[SourcePresentationHint] = None
    origin: Optional[str] = None
    sources: Optional[List["Source"]] = None
    adapterData: Optional[Any] = None
    checksums: Optional[List[Checksum]] = None

    def __post_init__(self):
        if not self.name and not self.path and not self.sourceReference:
            raise ValueError(
                f"Source requires either name, path, or source_reference. {self}"
            )

    @classmethod
    def create(cls, name: Optional[str] = None, path: Optional[str] = None, **kwargs):
        if path and not name:
            name = os.path.basename(path)

        return cls(name=name, path=path, **kwargs)


@dataclass(frozen=True)
class Breakpoint:
    verified: bool
    id: Optional[int] = None
    message: Optional[str] = None
    source: Optional[Source] = None
    line: Optional[int] = None
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None
    instructionReference: Optional[str] = None
    offset: Optional[int] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class SourceBreakpoint:
    line: int
    column: Optional[int] = None
    condition: Optional[str] = None
    hitCondition: Optional[str] = None
    logMessage: Optional[str] = None
    mode: Optional[str] = None


@dataclass(frozen=True)
class FunctionBreakpoint:
    name: str
    condition: Optional[str] = None
    hitCondition: Optional[str] = None


@dataclass(frozen=True)
class DataBreakpoint:
    dataId: str
    accessType: Optional[str] = None
    condition: Optional[str] = None
    hitCondition: Optional[str] = None


@dataclass(frozen=True)
class InstructionBreakpoint:
    instructionReference: str
    offset: Optional[int] = None
    condition: Optional[str] = None
    hitCondition: Optional[str] = None
    mode: Optional[str] = None


@dataclass(frozen=True)
class BreakpointLocation:
    line: int
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None


ColumnDescriptorType = Literal["string", "number", "boolean", "unixTimestampUTC"]


@dataclass(frozen=True)
class ColumnDescriptor:
    attributeName: str
    label: str
    format: Optional[str] = None
    type: Optional[ColumnDescriptorType] = None
    width: Optional[int] = None


@dataclass(frozen=True)
class ExceptionFilterOptions:
    filterId: str
    condition: Optional[str] = None


@dataclass(frozen=True)
class ExceptionPathSegment:
    names: List[str]
    negate: Optional[bool] = None


@dataclass(frozen=True)
class ExceptionOptions:
    breakMode: ExceptionBreakMode
    path: Optional[List[ExceptionPathSegment]] = None


@dataclass(frozen=True)
class ExceptionDetails:
    message: Optional[str] = None
    typeName: Optional[str] = None
    fullTypeName: Optional[str] = None
    evaluateName: Optional[str] = None
    stackTrace: Optional[str] = None
    innerException: Optional[List["ExceptionDetails"]] = None


@dataclass(frozen=True)
class ExceptionBreakpointsFilter:
    filter: str
    label: str
    description: Optional[str] = None
    default: Optional[bool] = None
    supportsCondition: Optional[bool] = None
    conditionDescription: Optional[str] = None


@dataclass(frozen=True)
class Thread:
    id: int
    name: str


@dataclass(frozen=True)
class StackFrame:
    id: int
    name: str
    line: int
    column: int
    source: Optional[Source] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None
    canRestart: Optional[bool] = None
    instructionPointerReference: Optional[str] = None
    moduleId: Optional[Union[int, str]] = None
    presentationHint: Optional[StackFramePresentationHint] = None


@dataclass(frozen=True)
class Scope:
    name: str
    variablesReference: int
    presentationHint: Optional[ScopePresentationHint] = None
    namedVariables: Optional[int] = None
    indexedVariables: Optional[int] = None
    expensive: bool = False
    source: Optional[Source] = None
    line: Optional[int] = None
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None


@dataclass(frozen=True)
class VariablePresentationHint:
    kind: Optional[VariablePresentationHintKind] = None
    attributes: Optional[List[str]] = None
    visibility: Optional[VariablePresentationHintVisibility] = None
    lazy: Optional[bool] = None


@dataclass(frozen=True)
class Variable:
    name: str
    value: str
    variablesReference: int
    type: Optional[str] = None
    presentationHint: Optional[VariablePresentationHint] = None
    evaluateName: Optional[str] = None
    namedVariables: Optional[int] = None
    indexedVariables: Optional[int] = None
    memoryReference: Optional[str] = None
    declarationLocationReference: Optional[int] = None
    valueLocationReference: Optional[int] = None

    @property
    def value_as_int(self):
        value = self.value
        # 'value' may have the variable value and summary.
        # Extract the variable value since summary can have nonnumeric characters.
        value = value.split(" ")[0]
        if value.startswith("0x"):
            return int(value, 16)
        elif value.startswith("0"):
            return int(value, 8)
        else:
            return int(value)


@dataclass(frozen=True)
class Message:
    id: int
    format: str
    variables: Optional[Dict[str, str]] = None
    sendTelemetry: Optional[bool] = None
    showUser: Optional[bool] = None
    url: Optional[str] = None
    urlLabel: Optional[str] = None


@dataclass(frozen=True)
class Module:
    id: str
    name: str
    path: Optional[str] = None
    isOptimized: Optional[bool] = None
    isUserCode: Optional[bool] = None
    version: Optional[str] = None
    symbolStatus: Optional[str] = None
    symbolFilePath: Optional[str] = None
    dateTimeStamp: Optional[str] = None
    addressRange: Optional[str] = None
    # custom
    debugInfoSize: Optional[str] = None


@dataclass(frozen=True)
class CompletionItem:
    label: str
    text: Optional[str] = None
    detail: Optional[str] = None
    start: Optional[int] = None
    length: int = 0

    def __repr__(self):
        # Use json as it is easier to see the diff on failure.
        return json.dumps(asdict(self), indent=4)

    def clone(self, **kwargs) -> "CompletionItem":
        """Creates a copy of this CompletionItem with specified fields modified."""
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True)
class ValueFormat:
    hex: Optional[bool] = None


@dataclass(frozen=True)
class StackFrameFormat:
    hex: Optional[bool] = None
    parameters: Optional[bool] = None
    parameterTypes: Optional[bool] = None
    parameterNames: Optional[bool] = None
    parameterValues: Optional[bool] = None
    line: Optional[bool] = None
    module: Optional[bool] = None
    includeAll: Optional[bool] = None


@dataclass(frozen=True)
class GotoTarget:
    id: int
    label: str
    line: int
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None
    instructionPointerReference: Optional[str] = None


@dataclass(frozen=True)
class StepInTarget:
    id: int
    label: str
    line: Optional[int] = None
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None


@dataclass(frozen=True)
class DisassembledInstruction:
    address: str
    instruction: str
    instructionBytes: Optional[str] = None
    symbol: Optional[str] = None
    location: Optional[Source] = None
    line: Optional[int] = None
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None
    presentationHint: Optional[str] = None


@dataclass(frozen=True)
class Capabilities:
    supportsConfigurationDoneRequest: Optional[bool] = None
    supportsFunctionBreakpoints: Optional[bool] = None
    supportsConditionalBreakpoints: Optional[bool] = None
    supportsHitConditionalBreakpoints: Optional[bool] = None
    supportsEvaluateForHovers: Optional[bool] = None
    exceptionBreakpointFilters: Optional[List[ExceptionBreakpointsFilter]] = None
    supportsStepBack: Optional[bool] = None
    supportsSetVariable: Optional[bool] = None
    supportsRestartFrame: Optional[bool] = None
    supportsGotoTargetsRequest: Optional[bool] = None
    supportsStepInTargetsRequest: Optional[bool] = None
    supportsCompletionsRequest: Optional[bool] = None
    completionTriggerCharacters: Optional[List[str]] = None
    supportsModulesRequest: Optional[bool] = None
    additionalModuleColumns: Optional[List[ColumnDescriptor]] = None
    supportedChecksumAlgorithms: Optional[List[ChecksumAlgorithm]] = None
    supportsRestartRequest: Optional[bool] = None
    supportsExceptionOptions: Optional[bool] = None
    supportsValueFormattingOptions: Optional[bool] = None
    supportsExceptionInfoRequest: Optional[bool] = None
    supportTerminateDebuggee: Optional[bool] = None
    supportSuspendDebuggee: Optional[bool] = None
    supportsDelayedStackTraceLoading: Optional[bool] = None
    supportsLoadedSourcesRequest: Optional[bool] = None
    supportsLogPoints: Optional[bool] = None
    supportsTerminateThreadsRequest: Optional[bool] = None
    supportsSetExpression: Optional[bool] = None
    supportsTerminateRequest: Optional[bool] = None
    supportsDataBreakpoints: Optional[bool] = None
    supportsReadMemoryRequest: Optional[bool] = None
    supportsWriteMemoryRequest: Optional[bool] = None
    supportsDisassembleRequest: Optional[bool] = None
    supportsCancelRequest: Optional[bool] = None
    supportsBreakpointLocationsRequest: Optional[bool] = None
    supportsClipboardContext: Optional[bool] = None
    supportsSteppingGranularity: Optional[bool] = None
    supportsInstructionBreakpoints: Optional[bool] = None
    supportsExceptionFilterOptions: Optional[bool] = None
    supportsSingleThreadExecutionRequests: Optional[bool] = None
    supportsDataBreakpointBytes: Optional[bool] = None
    breakpointModes: Optional[List[Any]] = None
    lldb_version: Optional[str] = field(
        metadata={"alias": "$__lldb_version"}, default=None
    )

    # lldb-dap custom capability.
    supportsModuleSymbolsRequest: Optional[bool] = None


@dataclass(frozen=True)
class ErrorResponse(Response):
    @dataclass(frozen=True)
    class Body:
        error: Optional[Message] = None

    message: Optional[str] = None
    body: Optional[Body] = None

    def __post_init__(self):
        assert not self.success, f"success field must be 'False' {self}"


@dataclass(frozen=True)
@args_protocol
class CancelArgs:
    requestId: Optional[int] = None
    progressId: Optional[str] = None

    command_ = "cancel"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
class RunInTerminalResponse(Response):
    @dataclass(frozen=True)
    class Body:
        processId: Optional[int] = None
        shellProcessId: Optional[int] = None

    body: Body


@dataclass(frozen=True)
class RunInTerminalArgs:
    # TODO: Fix this cwd is not optional
    cwd: Optional[str] = None
    args: List[str] = field(default_factory=list)
    kind: Optional[Literal["integrated", "external"]] = None
    title: Optional[str] = None
    env: Optional[Dict[str, Union[str, None]]] = None
    argsCanBeInterpretedByShell: Optional[bool] = None


@dataclass(frozen=True)
class RunInTerminalRequest(Request):
    arguments: RunInTerminalArgs


ReverseResponse = Union[RunInTerminalResponse, EmptyBodyResponse, ErrorResponse]
"""Possible Responses from a reverse Request"""


@dataclass(frozen=True)
class StartDebuggingRequestArgs:
    configuration: Dict[str, Any] = field(default_factory=dict)
    request: StartDebuggingRequestType = StartDebuggingRequestType.LAUNCH

    command_ = "startDebugging"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
class InitializeResponse(Response):
    body: Capabilities


@dataclass(frozen=True)
@args_protocol
class InitializeArgs:
    adapterID: str
    clientID: Optional[str] = None
    clientName: Optional[str] = None
    locale: Optional[str] = None
    linesStartAt1: Optional[bool] = None
    columnsStartAt1: Optional[bool] = None
    pathFormat: Optional[Literal["path", "uri"]] = None
    supportsVariableType: Optional[bool] = None
    supportsVariablePaging: Optional[bool] = None
    supportsRunInTerminalRequest: Optional[bool] = True
    supportsMemoryReferences: Optional[bool] = None
    supportsProgressReporting: Optional[bool] = None
    supportsInvalidatedEvent: Optional[bool] = None
    supportsMemoryEvent: Optional[bool] = None
    supportsArgsCanBeInterpretedByShell: Optional[bool] = None
    supportsStartDebuggingRequest: Optional[bool] = None
    supportsANSIStyling: Optional[bool] = None
    sourceInitFile: bool = field(
        metadata={"alias": "$__lldbSourceInitFile"}, default=False
    )

    command_ = "initialize"
    response_class_ = InitializeResponse


@dataclass(frozen=True)
class InitializedEvent(Event, event=EventName.INITIALIZED):
    event: Union[EventName, str] = field(metadata={"required": EventName.INITIALIZED})

    @dataclass(frozen=True)
    class Body:
        lldb_statistics: Dict[str, Any] = field(
            default_factory=dict, metadata={"alias": "$__lldb_statistics"}
        )

    body: Optional[Body] = None


@dataclass(frozen=True)
@args_protocol
class ConfigurationDoneArgs:
    command_ = "configurationDone"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class LaunchArgs:
    program: str
    noDebug: bool = False
    launchCommands: Optional[List[str]] = None
    cwd: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Union[Dict[str, str], List[str]]] = None
    detachOnError: Optional[bool] = None
    disableASLR: bool = False
    disableSTDIO: bool = False
    shellExpandArguments: bool = False
    console: Console = Console.INTERNAL
    stdio: Optional[List[Optional[str]]] = None

    # Configurations.
    debuggerRoot: Optional[str] = None
    enableAutoVariableSummaries: bool = False
    enableSyntheticChildDebugging: bool = False
    displayExtendedBacktrace: bool = False
    stopOnEntry: bool = False
    timeout: Optional[float] = None
    commandEscapePrefix: Optional[str] = None
    customFrameFormat: Optional[str] = None
    customThreadFormat: Optional[str] = None
    sourcePath: Optional[str] = None
    sourceMap: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    preInitCommands: Optional[List[str]] = None
    initCommands: Optional[List[str]] = None
    preRunCommands: Optional[List[str]] = None
    postRunCommands: Optional[List[str]] = None
    stopCommands: Optional[List[str]] = None
    exitCommands: Optional[List[str]] = None
    terminateCommands: Optional[List[str]] = None

    command_ = "launch"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class AttachArgs:
    @dataclass(frozen=True)
    class Session:
        targetId: int
        debuggerId: Optional[int] = None

    program: Optional[str] = None
    attachCommands: Optional[List[str]] = None
    pid: Optional[int] = None
    waitFor: Optional[bool] = None
    gdbRemotePort: Optional[int] = field(
        metadata={"alias": "gdb-remote-port"}, default=None
    )
    gdbRemoteHostname: Optional[str] = field(
        metadata={"alias": "gdb-remote-hostname"}, default=None
    )
    coreFile: Optional[str] = None
    session: Optional[Session] = None
    restart: Optional[Any] = field(metadata={"alias": "__restart"}, default=None)

    # Configurations.
    debuggerRoot: Optional[str] = None
    enableAutoVariableSummaries: Optional[bool] = None
    enableSyntheticChildDebugging: Optional[bool] = None
    displayExtendedBacktrace: Optional[bool] = None
    stopOnEntry: Optional[bool] = None
    timeout: Optional[float] = None
    commandEscapePrefix: Optional[str] = None
    customFrameFormat: Optional[str] = None
    customThreadFormat: Optional[str] = None
    sourcePath: Optional[str] = None
    sourceMap: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    preInitCommands: Optional[List[str]] = None
    initCommands: Optional[List[str]] = None
    preRunCommands: Optional[List[str]] = None
    postRunCommands: Optional[List[str]] = None
    stopCommands: Optional[List[str]] = None
    exitCommands: Optional[List[str]] = None
    terminateCommands: Optional[List[str]] = None

    command_ = "attach"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class RestartArgs:
    arguments: Optional[Union[LaunchArgs, AttachArgs]] = None

    command_ = "restart"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class DisconnectArgs:
    restart: Optional[bool] = None
    terminateDebuggee: Optional[bool] = None
    suspendDebuggee: Optional[bool] = None

    command_ = "disconnect"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class TerminateArgs:
    restart: Optional[bool] = None

    command_ = "terminate"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
class BreakpointLocationsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        breakpoints: List[BreakpointLocation]

    body: Body


@dataclass(frozen=True)
@args_protocol
class BreakpointLocationsArgs:
    source: Source
    line: int
    column: Optional[int] = None
    endLine: Optional[int] = None
    endColumn: Optional[int] = None

    command_ = "breakpointLocations"
    response_class_ = BreakpointLocationsResponse


@dataclass(frozen=True)
class AnyBreakpointsResponse(Response):
    """The response for 'setBreakpoints', 'setFunctionBreakpoints', 'setDataBreakpoints'
    and 'setInstructionBreakpoints'"""

    @dataclass(frozen=True)
    class Body:
        breakpoints: List[Breakpoint]

    body: Body


@dataclass(frozen=True)
@args_protocol
class SetBreakpointsArgs:
    source: Source
    breakpoints: Optional[List[SourceBreakpoint]] = None
    lines: Optional[List[int]] = None
    sourceModified: Optional[bool] = None

    command_ = "setBreakpoints"
    response_class_ = AnyBreakpointsResponse


@dataclass(frozen=True)
@args_protocol
class SetFunctionBreakpointsArgs:
    breakpoints: List[FunctionBreakpoint]

    command_ = "setFunctionBreakpoints"
    response_class_ = AnyBreakpointsResponse


@dataclass(frozen=True)
class SetExceptionBreakpointsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        breakpoints: Optional[List[Breakpoint]] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class SetExceptionBreakpointsArgs:
    filters: List[str]
    filterOptions: Optional[List[ExceptionFilterOptions]] = None
    exceptionOptions: Optional[List[ExceptionOptions]] = None

    command_ = "setExceptionBreakpoints"
    response_class_ = SetExceptionBreakpointsResponse


@dataclass(frozen=True)
@args_protocol
class DAPTestGetTargetBreakpointsArgs:
    command_ = "_testGetTargetBreakpoints"
    response_class_ = AnyBreakpointsResponse


@dataclass(frozen=True)
class DataBreakpointInfoResponse(Response):
    @dataclass(frozen=True)
    class Body:
        description: str
        dataId: Optional[str] = None
        accessTypes: Optional[List[str]] = None
        canPersist: Optional[bool] = None
        canBreakOnBytes: Optional[bool] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class DataBreakpointInfoArgs:
    name: str
    variablesReference: Optional[int] = None
    frameId: Optional[int] = None
    bytes: Optional[int] = None
    asAddress: Optional[bool] = None
    mode: Optional[str] = None

    command_ = "dataBreakpointInfo"
    response_class_ = DataBreakpointInfoResponse


@dataclass(frozen=True)
@args_protocol
class SetDataBreakpointsArgs:
    breakpoints: List[DataBreakpoint]

    command_ = "setDataBreakpoints"
    response_class_ = AnyBreakpointsResponse


@dataclass(frozen=True)
@args_protocol
class SetInstructionBreakpointsArgs:
    breakpoints: List[InstructionBreakpoint]

    command_ = "setInstructionBreakpoints"
    response_class_ = AnyBreakpointsResponse


@dataclass(frozen=True)
class CompileUnit:
    compileUnitPath: str


@dataclass(frozen=True)
class CompileUnitsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        compileUnits: List[CompileUnit]

    body: Body


@dataclass(frozen=True)
@args_protocol
class CompileUnitsArgs:
    moduleId: str

    command_ = "compileUnits"
    response_class_ = CompileUnitsResponse


@dataclass(frozen=True)
class ContinueResponse(Response):
    @dataclass(frozen=True)
    class Body:
        allThreadsContinued: Optional[bool] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class ContinueArgs:
    threadId: int = 0
    singleThread: Optional[bool] = None

    command_ = "continue"
    response_class_ = ContinueResponse


SteppingGranularity = Literal["statement", "line", "instruction"]


@dataclass(frozen=True)
@args_protocol
class NextArgs:
    threadId: int
    singleThread: Optional[bool] = None
    granularity: Optional[SteppingGranularity] = None

    command_ = "next"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class StepInArgs:
    threadId: int
    singleThread: Optional[bool] = None
    targetId: Optional[int] = None
    granularity: Optional[SteppingGranularity] = None

    command_ = "stepIn"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class StepOutArgs:
    threadId: int
    singleThread: Optional[bool] = None
    granularity: Optional[SteppingGranularity] = None

    command_ = "stepOut"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class GotoArgs:
    threadId: int
    targetId: int

    command_ = "goto"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
@args_protocol
class PauseArgs:
    threadId: int

    command_ = "pause"
    response_class_ = EmptyBodyResponse


@dataclass(frozen=True)
class StackTraceResponse(Response):
    @dataclass(frozen=True)
    class Body:
        stackFrames: List[StackFrame]
        totalFrames: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class StackTraceArgs:
    threadId: int
    startFrame: Optional[int] = None
    levels: Optional[int] = None
    format: Optional[StackFrameFormat] = None

    command_ = "stackTrace"
    response_class_ = StackTraceResponse


@dataclass(frozen=True)
class ScopesResponse(Response):
    @dataclass(frozen=True)
    class Body:
        scopes: List[Scope]

    body: Body


@dataclass(frozen=True)
@args_protocol
class ScopesArgs:
    frameId: int

    command_ = "scopes"
    response_class_ = ScopesResponse


@dataclass(frozen=True)
class VariablesResponse(Response):
    @dataclass(frozen=True)
    class Body:
        variables: List[Variable]

    body: Body


@dataclass(frozen=True)
@args_protocol
class VariablesArgs:
    variablesReference: int
    filter: Optional[Literal["indexed", "named"]] = None
    start: Optional[int] = None
    count: Optional[int] = None
    format: Optional[ValueFormat] = None

    command_ = "variables"
    response_class_ = VariablesResponse


@dataclass(frozen=True)
class SetVariableResponse(Response):
    @dataclass(frozen=True)
    class Body:
        value: str
        type: Optional[str] = None
        variablesReference: Optional[int] = None
        namedVariables: Optional[int] = None
        indexedVariables: Optional[int] = None
        memoryReference: Optional[str] = None
        valueLocationReference: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class SetVariableArgs:
    variablesReference: int
    name: str
    value: str
    format: Optional[ValueFormat] = None

    command_ = "setVariable"
    response_class_ = SetVariableResponse


@dataclass(frozen=True)
class SourceResponse(Response):
    @dataclass(frozen=True)
    class Body:
        content: str
        mimeType: Optional[str] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class SourceArgs:
    sourceReference: int
    source: Optional[Source] = None

    command_ = "source"
    response_class_ = SourceResponse


@dataclass(frozen=True)
class ThreadsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        threads: List[Thread]

    body: Body


@dataclass(frozen=True)
@args_protocol
class ThreadsArgs:
    command_ = "threads"
    response_class_ = ThreadsResponse


@dataclass(frozen=True)
class ModulesResponse(Response):
    @dataclass(frozen=True)
    class Body:
        modules: List[Module]
        totalModules: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class ModulesArgs:
    startModule: Optional[int] = None
    moduleCount: Optional[int] = None

    command_ = "modules"
    response_class_ = ModulesResponse


@dataclass(frozen=True)
class ModuleSymbol:
    """Mirrors the `Symbol` struct produced by lldb-dap's `moduleSymbols`
    request."""

    id: int
    isDebug: bool
    isSynthetic: bool
    isExternal: bool
    type: str
    fileAddress: int
    size: int
    name: str
    loadAddress: Optional[int] = None


@dataclass(frozen=True)
class ModuleSymbolsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        symbols: List[ModuleSymbol]
        totalSymbols: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class ModuleSymbolsArgs:
    moduleName: str
    moduleId: str = ""
    startIndex: Optional[int] = None
    count: Optional[int] = None

    command_ = "__lldb_moduleSymbols"
    response_class_ = ModuleSymbolsResponse


@dataclass(frozen=True)
class EvaluateResponse(Response):
    @dataclass(frozen=True)
    class Body:
        result: str
        variablesReference: int
        type: Optional[str] = None
        presentationHint: Optional[VariablePresentationHint] = None
        namedVariables: Optional[int] = None
        indexedVariables: Optional[int] = None
        memoryReference: Optional[str] = None
        valueLocationReference: Optional[int] = None

    body: Body


EvaluateContext = Literal["watch", "repl", "hover", "clipboard", "variables"]


@dataclass(frozen=True)
@args_protocol
class EvaluateArgs:
    expression: str
    frameId: Optional[int] = None
    context: Optional[EvaluateContext] = None
    format: Optional[ValueFormat] = None

    command_ = "evaluate"
    response_class_ = EvaluateResponse


@dataclass(frozen=True)
class StepInTargetsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        targets: List[StepInTarget]

    body: Body


@dataclass(frozen=True)
@args_protocol
class StepInTargetsArgs:
    frameId: int

    command_ = "stepInTargets"
    response_class_ = StepInTargetsResponse


@dataclass(frozen=True)
class GotoTargetsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        targets: List[GotoTarget]

    body: Body


@dataclass(frozen=True)
@args_protocol
class GotoTargetsArgs:
    source: Source
    line: int
    column: Optional[int] = None

    command_ = "gotoTargets"
    response_class_ = GotoTargetsResponse


@dataclass(frozen=True)
class CompletionsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        targets: List[CompletionItem]

    body: Body


@dataclass(frozen=True)
@args_protocol
class CompletionsArgs:
    text: str
    column: int
    frameId: Optional[int] = None
    line: Optional[int] = None

    command_ = "completions"
    response_class_ = CompletionsResponse


@dataclass(frozen=True)
class ExceptionInfoResponse(Response):
    @dataclass(frozen=True)
    class Body:
        exceptionId: str
        breakMode: ExceptionBreakMode
        description: Optional[str] = None
        details: Optional[ExceptionDetails] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class ExceptionInfoArgs:
    threadId: int

    command_ = "exceptionInfo"
    response_class_ = ExceptionInfoResponse


@dataclass(frozen=True)
class ReadMemoryResponse(Response):
    @dataclass(frozen=True)
    class Body:
        address: str
        unreadableBytes: Optional[int] = None
        data: Optional[str] = None  # base64-encoded bytes

    body: Body


@dataclass(frozen=True)
@args_protocol
class ReadMemoryArgs:
    memoryReference: str
    count: int
    offset: Optional[int] = None

    command_ = "readMemory"
    response_class_ = ReadMemoryResponse


@dataclass(frozen=True)
class WriteMemoryResponse(Response):
    @dataclass(frozen=True)
    class Body:
        offset: Optional[int] = None
        bytesWritten: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class WriteMemoryArgs:
    memoryReference: str
    data: str  # base64-encoded bytes
    offset: Optional[int] = None
    allowPartial: Optional[bool] = None

    command_ = "writeMemory"
    response_class_ = WriteMemoryResponse


@dataclass(frozen=True)
class DisassembleResponse(Response):
    @dataclass(frozen=True)
    class Body:
        instructions: List[DisassembledInstruction]

    body: Body


@dataclass(frozen=True)
@args_protocol
class DisassembleArgs:
    memoryReference: str
    instructionCount: int
    offset: Optional[int] = None
    instructionOffset: Optional[int] = None
    resolveSymbols: Optional[bool] = None

    command_ = "disassemble"
    response_class_ = DisassembleResponse


@dataclass(frozen=True)
class LocationsResponse(Response):
    @dataclass(frozen=True)
    class Body:
        source: Source
        line: int
        column: Optional[int] = None
        endLine: Optional[int] = None
        endColumn: Optional[int] = None

    body: Body


@dataclass(frozen=True)
@args_protocol
class LocationsArgs:
    locationReference: int

    command_ = "locations"
    response_class_ = LocationsResponse


@dataclass(frozen=True)
class StoppedEvent(Event, event=EventName.STOPPED):
    event: Union[EventName, str] = field(metadata={"required": EventName.STOPPED})

    @dataclass(frozen=True)
    class Body:
        reason: StoppedReason
        description: Optional[str] = None
        threadId: Optional[int] = None
        preserveFocusHint: Optional[bool] = None
        text: Optional[str] = None
        allThreadsStopped: Optional[bool] = None
        hitBreakpointIds: Optional[List[int]] = None

    body: Body


@dataclass(frozen=True)
class ContinuedEvent(Event, event=EventName.CONTINUED):
    event: Union[EventName, str] = field(metadata={"required": EventName.CONTINUED})

    @dataclass(frozen=True)
    class Body:
        threadId: int
        allThreadsContinued: Optional[bool] = None

    body: Body


@dataclass(frozen=True)
class ExitedEvent(Event, event=EventName.EXITED):
    event: Union[EventName, str] = field(metadata={"required": EventName.EXITED})

    @dataclass(frozen=True)
    class Body:
        exitCode: int

    body: Body


@dataclass(frozen=True)
class TerminatedEvent(Event, event=EventName.TERMINATED):
    event: Union[EventName, str] = field(metadata={"required": EventName.TERMINATED})

    @dataclass(frozen=True)
    class Body:
        restart: Optional[Any] = None
        lldb_statistics: Dict[str, Any] = field(
            default_factory=dict, metadata={"alias": "$__lldb_statistics"}
        )

    body: Optional[Body] = None


@dataclass(frozen=True)
class ThreadEvent(Event, event=EventName.THREAD):
    event: Union[EventName, str] = field(metadata={"required": EventName.THREAD})

    @dataclass(frozen=True)
    class Body:
        threadId: int
        reason: Literal["started", "exited"]

    body: Body


@dataclass(frozen=True)
class OutputEvent(Event, event=EventName.OUTPUT):
    event: Union[EventName, str] = field(metadata={"required": EventName.OUTPUT})

    @dataclass(frozen=True)
    class Body:
        output: str
        category: OutputCategory = OutputCategory.CONSOLE  # defaults to console
        group: Optional[Literal["start", "startCollapsed", "end"]] = None
        variablesReference: Optional[int] = None
        source: Optional[Source] = None
        line: Optional[int] = None
        column: Optional[int] = None
        data: Optional[Any] = None
        locationReference: Optional[int] = None

    body: Body


@dataclass(frozen=True)
class BreakpointEvent(Event, event=EventName.BREAKPOINT):
    event: Union[EventName, str] = field(metadata={"required": EventName.BREAKPOINT})

    @dataclass(frozen=True)
    class Body:
        reason: BreakpointReason
        breakpoint: Breakpoint

    body: Body


@dataclass(frozen=True)
class ModuleEvent(Event, event=EventName.MODULE):
    event: Union[EventName, str] = field(metadata={"required": EventName.MODULE})

    @dataclass(frozen=True)
    class Body:
        reason: ModuleReason
        module: Module

    body: Body


@dataclass(frozen=True)
class ProcessEvent(Event, event=EventName.PROCESS):
    event: Union[EventName, str] = field(metadata={"required": EventName.PROCESS})

    @dataclass(frozen=True)
    class Body:
        name: str
        systemProcessId: Optional[int] = None
        isLocalProcess: Optional[bool] = None
        startMethod: Optional[Literal["launch", "attach"]] = None
        pointerSize: Optional[int] = None

    body: Body


@dataclass(frozen=True)
class CapabilitiesEvent(Event, event=EventName.CAPABILITIES):
    event: Union[EventName, str] = field(metadata={"required": EventName.CAPABILITIES})

    @dataclass(frozen=True)
    class Body:
        capabilities: Capabilities

    body: Body


@dataclass(frozen=True)
class ProgressStartEvent(Event, event=EventName.PROGRESS_START):
    event: Union[EventName, str] = field(
        metadata={"required": EventName.PROGRESS_START}
    )

    @dataclass(frozen=True)
    class Body:
        progressId: str
        title: str
        requestId: Optional[int] = None
        cancellable: Optional[bool] = None
        message: Optional[str] = None
        percentage: Optional[float] = None

    body: Body


@dataclass(frozen=True)
class ProgressUpdateEvent(Event, event=EventName.PROGRESS_UPDATE):
    event: Union[EventName, str] = field(
        metadata={"required": EventName.PROGRESS_UPDATE}
    )

    @dataclass(frozen=True)
    class Body:
        progressId: str
        message: Optional[str] = None
        percentage: Optional[float] = None

    body: Body


@dataclass(frozen=True)
class ProgressEndEvent(Event, event=EventName.PROGRESS_END):
    event: Union[EventName, str] = field(metadata={"required": EventName.PROGRESS_END})

    @dataclass(frozen=True)
    class Body:
        progressId: str
        message: Optional[str] = None

    body: Body


@dataclass(frozen=True)
class InvalidatedEvent(Event, event=EventName.INVALIDATED):
    event: Union[EventName, str] = field(metadata={"required": EventName.INVALIDATED})

    @dataclass(frozen=True)
    class Body:
        areas: Optional[List[InvalidatedAreas]] = None
        threadId: Optional[int] = None
        stackFrameId: Optional[int] = None

    body: Body


@dataclass(frozen=True)
class MemoryEvent(Event, event=EventName.MEMORY):
    event: Union[EventName, str] = field(metadata={"required": EventName.MEMORY})

    @dataclass(frozen=True)
    class Body:
        memoryReference: str
        offset: int
        count: int

    body: Body
