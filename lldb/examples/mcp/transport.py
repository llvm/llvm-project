import asyncio
import dataclasses
import enum
import functools
import json
import logging
import pprint
import sys
import traceback
from typing import (
    Any,
    Awaitable,
    Generic,
    TypeVar,
    Callable,
    Union,
    Optional,
)

logger = logging.getLogger(__name__)


@enum.unique
class MessageType(enum.Enum):
    REQ = enum.auto()
    RESP = enum.auto()
    NOTE = enum.auto()


@dataclasses.dataclass(frozen=True, repr=False)
class Message:
    """Wrapper around the JSON payload of a MCP message."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Message":
        # Ensure the jsonrpc field is always set.
        payload["jsonrpc"] = "2.0"
        return cls(payload=payload)

    @classmethod
    def decode(cls, bytes: Union[str, bytes, bytearray]) -> "Message":
        # Ensure the jsonrpc field is always set.
        return cls(payload=json.loads(bytes))

    def __str__(self):
        return json.dumps(self.payload, indent=None, separators=(",", ":"))

    def __repr__(self):
        return "{}: {}".format(
            self.message_type.name.title(),
            pprint.pformat(self.payload, sort_dicts=False),
        )

    @functools.cached_property
    def message_type(self) -> MessageType:
        if "id" in self.payload and "method" in self.payload:
            return MessageType.REQ
        elif "id" in self.payload:
            return MessageType.RESP
        elif "method" in self.payload:
            return MessageType.NOTE
        assert False, f"Unknown message type: {self.payload}"

    def encode(self) -> bytes:
        msg = json.dumps(self.payload, indent=None, separators=(",", ":"))
        return f"{msg}\n".encode()

    def matches(self, other: Optional["Message"]) -> bool:
        """Returns true iff other is a subset of this message."""
        if not other:
            return True

        # The other payload must be a subset of this payload, meaning if we were to
        # add its payload to ours, the payload is the same.
        return self.payload | other.payload == self.payload

    # Various typed wrappers around self.payload['field']

    @property
    def method(self) -> str:
        return self.payload["method"]

    @property
    def id(self) -> int:
        return self.payload["id"]

    @property
    def params(self) -> dict[str, Any]:
        return self.payload.get("params", {})

    @property
    def result(self) -> dict[str, Any]:
        return self.payload.get("result", {})

    @property
    def error(self) -> dict[str, Any]:
        return self.payload.get("error", {})

    @property
    def success(self) -> bool:
        return not hasattr(self.payload, "error")


class Transport:
    r: asyncio.StreamReader
    w: asyncio.StreamWriter

    def __init__(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        self.r = r
        self.w = w

    def write(self, message: Message):
        logger.info("--> %s", message)
        self.w.write(message.encode())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.w.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.r.readline()
        if line == b"":
            raise StopAsyncIteration
        return Message.decode(line)


class Invoker:
    name: str
    message_type: MessageType
    defaults: dict
    transport: Optional[Transport] = None
    handler: Optional["MessageHandler"] = None

    def __init__(self, name, message_type: MessageType, defaults: dict = {}):
        self.name = name
        self.message_type = message_type
        self.defaults = defaults

    def __call__(self, **kwargs):
        assert self.transport and self.handler
        if self.message_type == MessageType.REQ:
            return self.handler.request(self.name, params=self.defaults | kwargs)
        elif self.message_type == MessageType.NOTE:
            return self.handler.event(self.name, params=self.defaults | kwargs)


class Handler:
    name: str
    message_type: MessageType

    def __init__(self, name: str, message_type: MessageType):
        self.name = name
        self.message_type = message_type

    def __call__(self):
        def wrap(fn):
            return RequestWrapper(self.name, fn)

        return wrap


Params = TypeVar("Params", bound=dict)
Result = TypeVar("Result")


class EventDescriptor(Generic[Params]):
    invoker: Callable[Params, None]
    handler: Callable[Params, None]

    def __init__(self, name: str):
        self.name = name

        self.invoker = Invoker(name=name, message_type=MessageType.NOTE)
        self.handler = Handler(name, MessageType.NOTE)


class RequestDescriptor(Generic[Params, Result]):
    invoker: Callable[Params, Awaitable[Result]]
    handler: Callable[Params, Awaitable[Result]]

    def __init__(self, name: str, defaults: Params = {}):
        self.name = name

        self.invoker = Invoker(name, MessageType.REQ, defaults)
        self.handler = Handler(name, MessageType.REQ)


class RequestWrapper:
    name: str
    fn: Callable
    handler: "MessageHandler"

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        assert self.handler is not None
        _ = kwargs.pop("_meta", None)
        return self.fn(self.handler, *args, **kwargs)


class MessageHandler:
    seq: int = 0
    handlers: dict[str, RequestWrapper] = {}
    invokers: dict[str, Invoker] = {}
    inflight: dict[int, asyncio.Future] = {}
    transport: Transport

    def __init_subclass__(cls):
        super().__init_subclass__()
        for i in dir(cls):
            attr = getattr(cls, i)
            if isinstance(attr, RequestWrapper):
                cls.handlers[attr.name] = attr
            if isinstance(attr, Invoker):
                cls.invokers[attr.name] = attr

    def __init__(self, transport: Transport):
        self.transport = transport
        for invoker in self.invokers.values():
            invoker.transport = transport
            invoker.handler = self
        for handlers in self.handlers.values():
            handlers.handler = self

    _handler: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self._handler = asyncio.create_task(self.run())
        return self

    async def run(self):
        async for message in self.transport:
            logger.info("<-- %s", message)
            if message.message_type == MessageType.REQ:
                handler = self.handlers.get(message.method)
                if not handler:
                    self.transport.write(
                        Message.from_dict(
                            {
                                "id": message.id,
                                "error": {
                                    "code": -32601,
                                    "message": "Method not found",
                                },
                            }
                        )
                    )
                    continue
                try:
                    result = await handler(**message.params)
                    self.transport.write(
                        Message.from_dict(
                            {
                                "id": message.id,
                                "result": result,
                            }
                        )
                    )
                except Exception as e:
                    print("Internal error:", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    self.transport.write(
                        Message.from_dict(
                            {
                                "id": message.id,
                                "error": {
                                    "code": -32603,
                                    "message": "Internal error",
                                },
                            }
                        )
                    )
            elif message.message_type == MessageType.RESP:
                future = self.inflight.pop(message.id, None)
                if not future:
                    continue
                future.set_result(message.result)
            elif message.message_type == MessageType.NOTE:
                fn = self.handlers.get(message.method)
                if fn:
                    fn(**message.params)
                else:
                    logger.info("no handler for %s", message.method)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._handler:
            self._handler.cancel()
            try:
                await self._handler
            except asyncio.CancelledError:
                pass
        self._handler = None

    async def request(self, name: str, params: dict):
        self.seq += 1
        msg = Message.from_dict(
            {
                "id": self.seq,
                "method": name,
                "params": params,
            }
        )
        resp_future = asyncio.get_running_loop().create_future()
        self.inflight[self.seq] = resp_future
        self.transport.write(msg)
        return await resp_future

    def event(self, name: str, params: dict):
        msg = Message.from_dict(
            {
                "method": name,
                "params": params,
            }
        )
        self.transport.write(msg)

    @classmethod
    async def acceptClient(
        cls,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        try:
            with Transport(reader, writer) as client:
                server = cls(client)
                await server.run()
        except:
            logger.exception("mcp client failed", exc_info=True)
