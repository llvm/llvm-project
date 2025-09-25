# lldb-mcp backport

A backport of the lldb-mcp protocol for older releases of lldb.

To load the backport use:

```
(lldb) command script import --allow-reload server.py
(lldb) start_mcp
```

Then you can use the `lldb-mcp` script in this directory to launch a client for
the running server.

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
