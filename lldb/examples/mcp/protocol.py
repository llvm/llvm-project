import os
import io
import sys
import json
import logging
import ctypes
import ctypes.util
from typing import TypedDict, Any, Literal, Optional
from transport import RequestDescriptor, EventDescriptor

logger = logging.getLogger(__name__)

PROC_PIDPATHINFO_MAXSIZE = 4 * 1024


def _is_valid_lldb_process(pid: int) -> bool:
    logger.info("checking if process %d is alive and is an lldb process", pid)
    try:
        # raises ProcessLookupError if pid does not exist.
        os.kill(pid, 0)
        if sys.platform == "darwin":
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            assert libc
            proc_pidpath = libc.proc_pidpath
            proc_pidpath.restype = ctypes.c_int
            proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            buf = ctypes.create_string_buffer(PROC_PIDPATHINFO_MAXSIZE)
            if proc_pidpath(pid, buf, PROC_PIDPATHINFO_MAXSIZE) <= 0:
                raise OSError(ctypes.get_errno())
            path = bytes(buf.value).decode()
            logger.info("path=%r", path)
            if "lldb" not in os.path.basename(path):
                logger.info("pid %d is invalid", pid)
                return False
        logger.info("pid %d is valid", pid)
        return True
    except ProcessLookupError:
        logger.info("pid %d is not alive", pid)
        return False
    except:
        logger.exception("failed to validate pid %d", pid)
        return False


class ServerInfo(TypedDict):
    connection_uri: str


def load() -> list[ServerInfo]:
    dir = os.path.expanduser("~/.lldb")
    contents = os.listdir(dir)
    server_infos = []
    for file in contents:
        if not file.startswith("lldb-mcp-") or not file.endswith(".json"):
            continue

        filename = os.path.join(dir, file)
        pid = int(file.removeprefix("lldb-mcp-").removesuffix(".json"))
        if not _is_valid_lldb_process(pid):
            # Process is dead, clean up the stale file.
            os.remove(filename)
            continue

        with open(filename) as f:
            server_infos.append(json.load(f))
    return server_infos


def cleanup():
    server_info_config = os.path.expanduser(f"~/.lldb/lldb-mcp-{os.getpid()}.json")
    if os.path.exists(server_info_config):
        os.remove(server_info_config)


def save(uri: str):
    server_info: ServerInfo = {"connection_uri": uri}
    with open(os.path.expanduser(f"~/.lldb/lldb-mcp-{os.getpid()}.json"), "w+") as f:
        json.dump(server_info, f)


class URI:
    scheme: str
    host: Optional[str]
    port: Optional[int]
    path: str

    def __init__(
        self,
        *,
        scheme="",
        host: Optional[str] = None,
        port: Optional[int] = None,
        path="",
    ):
        self.scheme = scheme
        self.host = host
        self.port = port
        self.path = path

    @classmethod
    def parse(cls, input: str) -> "URI":
        assert ":" in input
        uri = URI()
        uri.scheme, rest = input.split(":", maxsplit=1)
        assert uri.scheme.isascii()
        if rest.startswith("//"):
            rest = rest.removeprefix("//")
            if "/" in rest:
                uri.host, rest = rest.split("/", maxsplit=1)
            else:
                uri.host = rest
                rest = ""
        uri.path = rest
        if uri.host is not None and ":" in uri.host:
            uri.host, raw_port = uri.host.rsplit(":", maxsplit=1)
            assert raw_port.isdigit()
            uri.port = int(raw_port)
        return uri

    def append(self, path: str) -> "URI":
        return URI(
            scheme=self.scheme,
            host=self.host,
            port=self.port,
            path=os.path.join(self.path, path),
        )

    def __str__(self):
        os = io.StringIO()
        os.write(self.scheme)
        os.write(":")
        if self.host or self.port:
            os.write("//")
            if self.host:
                os.write(self.host)
            if self.port:
                os.write(":")
                os.write(self.port)
        if self.path and self.path != "/":
            os.write(self.path)
        return os.getvalue()


class ImplementationVersion(TypedDict):
    name: str
    version: str


class Tool(TypedDict):
    name: str
    title: str
    description: str
    inputSchema: dict


class Resource(TypedDict):
    uri: str
    name: str


class ListToolsResult(TypedDict):
    tools: list[Tool]


class CallToolParams(TypedDict):
    name: str
    arguments: Any


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class CallToolResult(TypedDict):
    content: list[TextContent]
    isError: bool


class ComponentCapabilities(TypedDict, total=False):
    listChanged: bool
    subscribe: bool


class ServerCapabilities(TypedDict):
    tools: ComponentCapabilities


class InitializeParams(TypedDict):
    capabilities: dict
    clientInfo: ImplementationVersion
    protocolVersion: str


class InitializeResult(TypedDict):
    capabilities: ServerCapabilities
    protocolVersion: str
    serverInfo: ImplementationVersion


initialize = RequestDescriptor[InitializeParams, InitializeResult](
    "initialize",
    defaults={
        "protocolVersion": "2024-11-05",
        "clientInfo": {
            "name": "lldb-mcp",
            "version": "0.0.1",
        },
        "capabilities": {
            "roots": {"listChanged": True},
            "sampling": {},
            "elicitation": {},
        },
    },
)
initialized = EventDescriptor[None](name="initialized")
toolsList = RequestDescriptor[None, ListToolsResult](name="tools/list")
toolsCall = RequestDescriptor[CallToolParams, CallToolResult](name="tools/call")
