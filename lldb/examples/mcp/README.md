# lldb-mcp backport

A backport of the lldb-mcp protocol for older releases of lldb.

To load the backport use:

```
(lldb) command script import --allow-reload server.py
(lldb) start_mcp
```

Then you can use the `./lldb-mcp` script in this directory to launch a client
for the running server.

For example,

```json
{
  "mcpServers": {
    "lldb": {
      "command": "<path>/lldb-mcp",
      "args": ["--log-file=/tmp/lldb-mcp.log", "--timeout=30.0"]
    }
  }
}
```

## Development

For getting started with making changes to this backport, use the
[MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) to run the
binary.

In one terminal, start the lldb server:

```
$ lldb
(lldb) command script import --allow-reload server.py
(lldb) start_mcp --log-file=/tmp/lldb-mcp-server.log
```

Then launch the inspector to run specific operations.

```sh
$ npx @modelcontextprotocol/inspector ./lldb-mcp --log-file=/tmp/lldb-mcp.log
```
