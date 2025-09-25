"""
An implementation of the lldb-mcp server.
"""

from typing import Any, Optional
import argparse
import asyncio
import lldb
import logging
import protocol
import queue
import pathlib
import traceback
import shlex
import threading
import transport

logger = logging.getLogger(__name__)


SCHEME = "lldb-mcp"
DEBUGGER_HOST = "debugger"
BASE_DEBUGGER_URI = protocol.URI(scheme=SCHEME, host=DEBUGGER_HOST, path="/")


class Tool:
    name: str
    title: str
    description: str
    inputSchema: dict

    def to_protocol(self) -> protocol.Tool:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }

    async def call(self, **kwargs) -> protocol.CallToolResult:
        assert False, "Implement in a subclass."


class CommandTool(Tool):
    name = "command"
    title = "LLDB Command"
    description = "Evaluates an lldb command."
    inputSchema = {
        "type": "object",
        "properties": {
            "debugger": {
                "type": "string",
                "description": "The debugger ID or URI to a specific debug session. If not specified, the first debugger will be used.",
            },
            "command": {
                "type": "string",
                "description": "An lldb command to run.",
            },
        },
    }

    async def call(
        self, *, command: Optional[str] = None, debugger: Optional[str] = None, **kwargs
    ) -> protocol.CallToolResult:
        if debugger:
            if debugger.isdigit():
                id = int(debugger)
            else:
                logger.info("Parsing %s", debugger)
                uri = protocol.URI.parse(debugger)
                logger.info("Parsed URI: %s", uri)
                assert uri.scheme == SCHEME
                assert uri.host == DEBUGGER_HOST
                raw_id = uri.path.removeprefix("/")
                assert raw_id.isdigit()
                id = int(raw_id)
            dbg_inst = lldb.SBDebugger.FindDebuggerWithID(id)
        else:
            for i in range(100):
                dbg_inst = lldb.SBDebugger.FindDebuggerWithID(i)
                if dbg_inst.IsValid():
                    break
        assert dbg_inst.IsValid()
        result = lldb.SBCommandReturnObject()
        dbg_inst.GetCommandInterpreter().HandleCommand(command, result)
        contents: list[protocol.TextContent] = []
        if result.GetOutputSize():
            contents.append({"type": "text", "text": str(result.GetOutput())})
        if result.GetErrorSize():
            contents.append({"type": "text", "text": str(result.GetError())})
        return {
            "content": contents,
            "isError": not result.Succeeded(),
        }


class DebuggerList(Tool):
    name = "debugger_list"
    title = "List Debuggers"
    description = "List debuggers associated with this server."
    inputSchema = {"type": "object"}

    async def call(self, **_kwargs) -> protocol.CallToolResult:
        out = ""

        for i in range(100):
            debugger = lldb.SBDebugger.FindDebuggerWithID(i)
            if debugger.IsValid():
                uri = BASE_DEBUGGER_URI.append(str(i))
                out += f"- {uri}\n"

        return {
            "content": [
                {"type": "text", "text": out},
            ],
            "isError": False,
        }


class MCPServer(transport.MessageHandler):
    tools: dict[str, Tool]

    def __init__(
        self, transport: transport.Transport, tools=[CommandTool(), DebuggerList()]
    ):
        super().__init__(transport)
        self.tools = {tool.name: tool for tool in tools}

    def __del__(self):
        print("deleting MCPServer....")

    @protocol.initialize.handler()
    async def initialize(
        self, **params: protocol.InitializeParams
    ) -> protocol.InitializeResult:
        return protocol.InitializeResult(
            capabilities={"tools": {"listChanged": True}},
            protocolVersion="2024-11-05",
            serverInfo={"name": "lldb-mcp", "version": "0.0.1"},
        )

    @protocol.initialized.handler()
    def initialized(self):
        print("Client initialized...")

    @protocol.toolsList.handler()
    async def listTools(self) -> protocol.ListToolsResult:
        return {"tools": [tool.to_protocol() for tool in self.tools.values()]}

    @protocol.toolsCall.handler()
    async def callTool(
        self, name: str, arguments: Optional[Any] = None
    ) -> protocol.CallToolResult:
        tool = self.tools[name]
        if arguments is None:
            arguments = {}
        return await tool.call(**arguments)


server: Optional[asyncio.AbstractServer] = None


def get_parser():
    parser = argparse.ArgumentParser("lldb-mcp")
    parser.add_argument("-l", "--log-file", type=pathlib.Path)
    parser.add_argument("-t", "--timeout", type=float, default=30.0)
    parser.add_argument("connection", nargs="?", default="listen://[127.0.0.1]:0")
    return parser


async def run(opts: argparse.Namespace, notify: Optional[queue.Queue] = None):
    global server
    conn: str = opts.connection
    assert conn.startswith("listen://"), "Invalid connection specifier"
    hostname, port = conn.removeprefix("listen://").split(":")
    hostname = hostname.removeprefix("[").removesuffix("]")

    logging.basicConfig(filename=opts.log_file, level=logging.DEBUG, force=True)

    server = await asyncio.start_server(MCPServer.acceptClient, hostname, int(port))
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    if notify:
        notify.put(addrs)
    else:
        print(f"Serving on {addrs}")

    sock_name = server.sockets[0].getsockname()
    (h, p) = sock_name[0], sock_name[1]
    protocol.save(f"connection://[{h}]:{p}")

    async with server:
        await server.serve_forever()


# A registration count, if this module is loaded for multiple debugger then we
# should only stop the global server if all registrations have been removed
registration_count = 0


def stop():
    """Stop the server, if one exists."""
    global server
    protocol.cleanup()
    if not server:
        return
    server.close()  # Stop accepting new connections
    server = None


class CommandStart:
    # The CommandStart is being used to track when the interpreter exits. lldb
    # does not call `Py_Finalize()`, so `atexit` calls are never invoked. In
    # order to ensure we shutdown the server and clean up the server info
    # records we use the `__del__` method to trigger the clean up as a best
    # effort attempt at a clean shutdown.
    def __init__(self, debugger, internal_dict):
        global registration_count
        registration_count += 1

    def __del__(self):
        global registration_count
        registration_count -= 1
        if registration_count == 0:
            stop()

    def __call__(self, debugger, command, exe_ctx, result):
        """Start an MCP server in a background thread."""
        global server

        if server is not None:
            print("Server already running.", file=result)
            return

        command_args = shlex.split(command)
        opts = get_parser().parse_args(command_args)

        print("Starting LLDB MCP Server...", file=result)

        notify = queue.Queue()

        def start_server():
            asyncio.run(run(opts, notify))

        thr = threading.Thread(target=start_server)
        thr.start()

        addrs = notify.get()
        print(f"Serving on {addrs}", file=result)
        result.SetStatus(lldb.eReturnStatusSuccessFinishNoResult)


def lldb_stop(debugger, command, exe_ctx, result, internal_dict):
    """Stop an MCP server."""
    global server
    try:
        if server is None:
            print("Server is stopped.", file=result)
            result.SetStatus(lldb.eReturnStatusSuccessFinishNoResult)
            return

        print("Server stopping...", file=result)
        stop()
        print("Server stopped.", file=result)

        result.SetStatus(lldb.eReturnStatusSuccessFinishNoResult)
    except:
        logging.exception("failed to stop MCP server")
        traceback.print_exc(file=result)
        result.SetStatus(lldb.eReturnStatusFailed)


def __lldb_init_module(
    debugger: lldb.SBDebugger,
    internal_dict: dict[Any, Any],
) -> None:
    debugger.HandleCommand("command script add -o -c server.CommandStart start_mcp")
    debugger.HandleCommand("command script add -o -f server.lldb_stop stop_mcp")
    print("Registered command 'start_mcp' and 'stop_mcp'.")
