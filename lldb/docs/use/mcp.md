# Model Context Protocol (MCP)

LLDB supports the [Model Context Protocol](https://modelcontextprotocol.io)
(MCP). This structured, machine-friendly protocol allows AI models to access
and interact with external tools, for example debuggers. Using MCP, an AI agent
can execute LLDB commands to control the debugger: set breakpoints, inspect
memory, step through code. This can range from helping you run a specific
command you cannot immediately remember, to a fully agent-driven debugging
experience.

## Getting Started

LLDB ships with `lldb-mcp`, a binary that speaks MCP over standard input and
output (stdio). Point your MCP client at it, and you are all set.

Configuration example for [Claude Code](https://modelcontextprotocol.io/quickstart/user):

```
claude mcp add lldb --transport stdio -- /path/to/lldb-mcp
```

Configuration example (`mcp.json`) for [Visual Studio Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers):

```json
{
  "servers": {
    "lldb": {
      "type": "stdio",
      "command": "/path/to/lldb-mcp"
    }
  }
}
```

The MCP client launches one `lldb-mcp` process per connection and shuts it down
when it disconnects, taking any session it created with it.

## Tools

Tools are a primitive in the Model Context Protocol that enable servers to
expose functionality to clients. `lldb-mcp` exposes four.

### `session_create`

Creates a new debug session and returns its URI. This is equivalent to
launching a new instance of `lldb` on the command line. Sessions look like
this:

```
lldb-mcp://instance/{pid}/debugger/{id}
```

The `pid` identifies the process hosting the session and the `id` identifies
the debugger inside it. Pass the whole URI back to the other tools.

### `command`

Runs an LLDB command in a debug session and returns its output, the same text
you would see in the LLDB command interpreter. It takes:

- `command` (required): the command to run, for example `breakpoint set --name main`.
- `debugger` (optional): the URI of the session to run it in. When omitted, the
  command runs in the first session `lldb-mcp` created.

Commands run one at a time and the result comes back when the command finishes.

### `sessions_list`

Lists every debug session reachable from this `lldb-mcp`, one URI per line.
That includes sessions it created itself and sessions in LLDB instances running
elsewhere on the machine (see [Attaching to a Running LLDB](#attaching-to-a-running-lldb)).

### `session_close`

Closes a session and frees its resources. It takes a single required `session`
argument, the URI to close. Only sessions that `lldb-mcp` created can be closed
this way. An interactive LLDB that a person is using belongs to that person, so
closing it is refused.

## A Typical Session

Creating a session, debugging in it, and cleaning up looks like this:

```
session_create                            -> lldb-mcp://instance/4711/debugger/1
command "target create /tmp/hello"        -> Current executable set to '/tmp/hello' (arm64).
command "breakpoint set --name add"       -> Breakpoint 1: 4 locations.
command "run"                             -> Process 4713 stopped
                                             * thread #1, stop reason = breakpoint 1.1
                                                 frame #0: hello`add(a=2, b=3) at hello.c:2
command "frame variable"                  -> (int) a = 2
                                             (int) b = 3
command "continue"                        -> Process 4713 exited with status = 0
session_close                             -> deleted lldb-mcp://debugger/1
```

:::{note}
Sessions start in asynchronous mode, where `run` and `continue` return before
the process actually stops. Commands that need a stopped process then fail with
"Command requires a process which is currently stopped". Run
`script lldb.debugger.SetAsync(False)` once, right after `session_create`, to
get the synchronous behavior shown above.
:::

The debuggee's own output does not come back through MCP. Only debugger output
does. Redirect the program's output to a file and read it back if you need it.

## Attaching to a Running LLDB

Besides the sessions it creates, `lldb-mcp` can drive LLDB instances you are
already using, so an agent can inspect and steer the exact session you have in
front of you.

In that LLDB, start an MCP server:

```
(lldb) protocol-server start MCP
MCP server started with connection listeners: connection://[::1]:59999, connection://[127.0.0.1]:59999
```

The server picks a free port on localhost by default. To listen somewhere
specific, pass a URI, either `listen://[host]:port` for TCP or
`accept:///path/to/socket` for a Unix domain socket:

```
(lldb) protocol-server start MCP listen://localhost:59999
```

The server stops when LLDB exits, or explicitly:

```
(lldb) protocol-server stop MCP
```

`protocol-server get MCP` reports where a running server is listening. Starting
a server when one is already running, or stopping one that is not, is an error.

Once the server is up, that LLDB's sessions show up in `sessions_list` and
accept `command`, exactly like sessions `lldb-mcp` created. You do not need to
configure a port anywhere: each LLDB with a running MCP server records itself in
`~/.lldb`, and `lldb-mcp` finds it there.

:::{note}
Discovery happens once, when `lldb-mcp` starts. An LLDB you launch afterwards is
not picked up until the client reconnects to the MCP server, which usually means
restarting or reloading the MCP server in your client.
:::

## Resources

Resources are a primitive in the Model Context Protocol that allow servers to
expose content that can be read by clients. `lldb-mcp` exposes one resource per
debugger and one per target, across every session it can reach.

Debugger resources use the following URI:

```
lldb://instance/<pid>/debugger/<debugger id>
```

Example output:

```json
{
  "contents": [
    {
      "uri": "lldb://instance/4711/debugger/1",
      "mimeType": "application/json",
      "text": "{\"debugger_id\":1,\"name\":\"debugger_1\",\"num_targets\":1}"
    }
  ]
}
```

Debuggers can contain one or more targets, which are accessible using the
following URI:

```
lldb://instance/<pid>/debugger/<debugger id>/target/<target idx>
```

Example output:

```json
{
  "contents": [
    {
      "uri": "lldb://instance/4711/debugger/1/target/0",
      "mimeType": "application/json",
      "text": "{\"arch\":\"arm64-apple-macosx26.0.0\",\"debugger_id\":1,\"dummy\":false,\"path\":\"/tmp/hello\",\"platform\":\"host\",\"selected\":true,\"target_idx\":0}"
    }
  ]
}
```

Note that unlike the debugger id, which is unique, the target index is not
stable and may be reused when a target is removed and a new target is added.

## Troubleshooting

**"no debugger found" from `command`.** There is no session to run the command
in. Call `session_create` first, or pass the URI of an existing session.

**"Command requires a process which is currently stopped".** The session is in
asynchronous mode. Run `script lldb.debugger.SetAsync(False)` in it.

**"can only close sessions that lldb-mcp created".** `session_close` refuses to
tear down an interactive LLDB. Quit that LLDB yourself.

**A running LLDB does not show up in `sessions_list`.** Either its MCP server is
not running, which `protocol-server get MCP` will tell you, or it started after
`lldb-mcp` did. Restart the MCP server in your client to rediscover.

To see the JSON-RPC traffic between your client and `lldb-mcp`, set
`LLDB_MCP_LOG` in the environment. Messages are written to stderr, since stdout
carries the protocol.

The MCP server inside LLDB logs to the `Host` log channel:

```
(lldb) log enable lldb host
```

## Implementation

This section covers how the pieces fit together, for those working on LLDB
itself.

`lldb-mcp` is a multiplexer. It presents a single MCP server to the client and
fans out to one or more backends, each an LLDB MCP server reached over a socket
and identified by the pid of the process hosting it.

```
                           ┌──────────┐
                           │   LLDB   │
                           └────┬─────┘
                                │ socket
                                │
┌──────────┐              ┌─────┴─────┐              ┌──────────┐
│ in-proc  ├────socket────┤  lldb-mcp ├─────stdio────┤MCP Client│
│  LLDB    │              └─────┬─────┘              └──────────┘
└──────────┘                    │ socket
                                │
                           ┌────┴─────┐
                           │   LLDB   │
                           └──────────┘
```

There are two kinds of backend. The **local** backend is an MCP server that
`lldb-mcp` starts inside its own process, through `SBProtocolServer`. The
sessions it hosts are the ones `session_create` makes. **Remote** backends are
the separate LLDB processes discovered through the registry. Both are driven the
same way, over a socket through an `mcp::Client`, which keeps the tool
implementations in one place rather than special-casing the in-process path.

Requests are dispatched three ways. `initialize` and `tools/list` are answered
by the multiplexer directly. `sessions_list` and `resources/list` fan out to
every live backend and aggregate, keyed by pid so output is deterministic. A
backend that fails or has disconnected is omitted rather than failing the whole
listing. `command`, `resources/read`, and `session_close` are routed to a single
backend by the pid parsed out of the URI.

Backends only know their own local `lldb-mcp://debugger/{id}` and
`lldb://debugger/{id}` URIs. The multiplexer rewrites them into the
instance-qualified form in both directions, so a client never sees an ambiguous
id and a backend never sees a pid it does not understand.

`session_create` and `session_close` map onto the `debugger_create` and
`debugger_delete` tools on the local backend. Session ownership is enforced by
comparing the pid in the URI against the local backend's, which is why closing
someone else's session is refused. Running a command in one is not: any
`lldb-mcp` on the machine can drive any discovered session.

### Discovery

An LLDB that starts an MCP server writes `~/.lldb/lldb-mcp-<pid>.json`,
recording the pid and the URI to connect to. The entry is written only once the
server is listening, and removed on a clean exit. `lldb-mcp` reads the directory
at startup and connects to each entry, pruning any that fails to connect, since
that means the instance died without cleaning up. `lldb-mcp` registers itself
too, so its managed sessions are visible to other `lldb-mcp` processes.

### Adding Tools and Resources

The tool and resource-provider set lives in
`lldb/source/Plugins/Protocol/MCP/` and is installed by
`lldb_private::mcp::PopulateServer`. Sharing one installer keeps every MCP
server consistent, whether it runs in the plugin or is hosted in-process by an
embedder. Adding a tool means subclassing `lldb_protocol::mcp::Tool`, and adding
a resource means subclassing `lldb_protocol::mcp::ResourceProvider`, then
registering it there.

A tool added this way is exposed by the LLDB MCP server, not automatically by
`lldb-mcp`. Because the multiplexer owns the client-facing surface, it also
needs a case in `HandleToolsList` and `HandleToolsCall`, plus a routing decision
if its arguments carry a URI.

Note that the protocol version LLDB implements is `2024-11-05`, which has no
structured content. Tools return their output as text.
