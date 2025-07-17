# Model Context Protocol (MCP)

LLDB supports the [Model Context Protocol](https://modelcontextprotocol.io)
(MCP). This structured, machine-friendly protocol allows AI models to access
and interact with external tools, for example debuggers. Using MCP, an AI agent
can execute LLDB commands to control the debugger: set breakpoints, inspect
memory, step through code. This can range from helping you run a specific
command you cannot immediately remember, to a fully agent-driven debugging
experience.

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

The commands will fail if a server is already running or not running
respectively.

## MCP Client

MCP uses standard input/output (stdio) for communication between client and
server. The exact configuration depends on the client, but most applications
allow you to specify an MCP server as a binary and arguments. This means that
you need to use something like `netcat` to connect to LLDB's MCP server and
forward communication over stdio over the network connection.

```
┌──────────┐               ┌──────────┐               ┌──────────┐
│          │               │          │               │          │
│   LLDB   ├─────socket────┤  netcat  ├─────stdio─────┤MCP Client│
│          │               │          │               │          │
└──────────┘               └──────────┘               └──────────┘
```

Configuration example for [Claude Code](https://modelcontextprotocol.io/quickstart/user):

```json
{
  "mcpServers": {
    "tool": {
      "command": "/usr/bin/nc",
      "args": ["localhost", "59999"]
    }
  }
}
```

Configuration example for [Visual Studio Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers):

```json
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

## Tools

Tools are a primitive in the Model Context Protocol that enable servers to
expose functionality to clients.

LLDB's MCP integration exposes one tool, named `lldb_command` which allows the
model to run the same commands a user would type in the LLDB command
interpreter. It takes two arguments:

1. The unique debugger ID as a number.
2. The command and its arguments as a string.

## Resources

Resources are a primitive in the Model Context Protocol that allow servers to
expose content that can be read by clients.

LLDB's MCP integration exposes a resource for each debugger and target
instance. Debugger resources are accessible using the following URI:

```
lldb://debugger/<debugger id>
```

Example output:

```json
{
  "contents": [
    {
      "uri": "lldb://debugger/1",
      "mimeType": "application/json",
      "text": "{\"debugger_id\":1,\"name\":\"debugger_1\",\"num_targets\":1}"
    }
  ]
}
```

Debuggers can contain one or more targets, which are accessible using the
following URI:

```
lldb://debugger/<debugger id>/target/<target idx>
```

Example output:

```json
{
  "contents": [
    {
      "uri": "lldb://debugger/1/target/0",
      "mimeType": "application/json",
      "text": "{\"arch\":\"arm64-apple-macosx26.0.0\",\"debugger_id\":1,\"dummy\":false,\"path\":\"/bin/count\",\"platform\":\"host\",\"selected\":true,\"target_idx\":0}"
    }
  ]
}
```

Note that unlike the debugger id, which is unique, the target index is not
stable and may be reused when a target is removed and a new target is added.

## Troubleshooting

The MCP server uses the `Host` log channel. You can enable logging with the
`log enable` command.

```
(lldb) log enable lldb host
```
