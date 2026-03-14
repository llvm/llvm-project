This directory contains:
- the XPC transport layer (alternative transport layer to JSON-RPC)
- XPC framework wrapper that wraps around Clangd to make it a valid XPC service
- XPC test-client

MacOS only. Feature is guarded by CLANGD_BUILD_XPC, including whole xpc/ dir.