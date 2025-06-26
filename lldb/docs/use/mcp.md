# Model Context Protocol (MCP)

LLDB supports for the [Model Context Protocol][1] (MCP). This structured,
machine-friendly protocol allows AI models to access and interact with external
tools, such as the debugger. Using MCP, an AI agent can execute LLDB commands
to control the debugger: set breakpoints, inspect memory, step through code.
This can range from helping you run a specific command you can't immediately
remember to a fully agent-driven debugging experiences

## MCP Server

To start the MCP server in LLDB, use the `protocol-server start` command.
Specify `MCP` as the protocol and provide a URI to listen on. For example, to
start listening for local TCP connections on port `59999`, use the following
command:

```
(lldb) protocol-server start MCP listen://localhost:59999
MCP server started with connection listeners: connection://[::1]:59999, connection://[127.0.0.1]:59999
```

The server will automatically stop when exiting LLDB, or it can be stopped
explicitly with the `protocol-server stop` command.

```
(lldb) protocol-server stop MCP
```

## MCP Client

MCP uses standard input/output (stdio) for communication between client and
server. The exact configuration depends on the client, but most applications
allow you to specify an MCP server as a binary and arguments. This means that
you need to use something like `netcat` to connect to LLDB's MCP server and
forward communication over stdio over the network connection.

Configuration example for [Claude Code][2]:

```
{
  "mcpServers": {
    "tool": {
      "command": "/usr/bin/nc",
      "args": ["localhost", "59999"]
    }
  }
}
```

Configuration example for [Visual Studio Code][3]:

```
{
  "mcp": {
    "servers": {
      "lldb": {
        "type": "stdio",
        "command": "/usr/bin/nc",
        "args": ["localhost", "59999"]
      }
    }
  }
}
```

[1]: https://modelcontextprotocol.io
[2]: https://modelcontextprotocol.io/quickstart/user
[3]: https://code.visualstudio.com/docs/copilot/chat/mcp-servers
