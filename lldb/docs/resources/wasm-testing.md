# Testing LLDB with WebAssembly

The LLDB API test suite can compile its test programs to WebAssembly and debug
them through a standalone Wasm runtime. The runtime executes the module and
serves a GDB remote stub. LLDB's `wasm` platform launches the runtime, picks a
free TCP port and connects to the stub on it.

This page covers configuring LLDB and the `wasm` platform. For building a
runtime, see the runtime's own documentation.

## Prerequisites

- A [wasi-sdk](https://github.com/WebAssembly/wasi-sdk) release, which provides
  the WASI sysroot, the wasm `compiler-rt` builtins and a `wasm-ld`.
- A Wasm runtime that serves a GDB remote stub. WAMR and WasmKit are shown
  below.

## Choosing a compiler

Test programs are built for `wasm32-wasip1`, which both the wasi-sdk clang and
an upstream in-tree clang can target. `LLDB_TEST_COMPILER` picks one. It
defaults to the in-tree `clang` whenever clang is part of the build.

Prefer the in-tree clang, since it matches the LLDB under test and exercises
in-tree DWARF and codegen changes. Unlike the wasi-sdk clang, it needs two
things supplied:

- `wasm-ld`. Enable LLD in the build and it lands next to `clang`, where the
  driver looks before `PATH`. Without LLD, symlink the wasi-sdk's `wasm-ld` into
  that same directory.
- The wasm `compiler-rt` builtins, through `LLDB_TEST_RESOURCE_DIR` pointed at
  the wasi-sdk's clang resource directory.

Either compiler needs the WASI sysroot, through `LLDB_TEST_SYSROOT`.

## Configuring the platform

The `wasm` platform is selected with `--platform-name wasm`. Four global
settings describe how to invoke the runtime:

| Setting | Meaning |
| --- | --- |
| `platform.plugin.wasm.runtime-path` | Path to the runtime binary. A name without a directory separator is looked up in `PATH`. |
| `platform.plugin.wasm.port-arg` | Argument carrying the GDB remote port. LLDB concatenates the port it chose. |
| `platform.plugin.wasm.env-arg` | Argument forwarding one environment variable. LLDB concatenates a `key=value` pair, once per variable in the inferior's environment. When empty, no environment is forwarded. |
| `platform.plugin.wasm.runtime-args` | Extra arguments for the runtime. They precede the port argument. |

LLDB assembles the runtime's command line as:

```
<runtime-path> <runtime-args...> <port-arg><port> <env-arg><key=value>... <module> <inferior-args...>
```

`runtime-args` precedes the port argument, so a runtime that dispatches on a
leading subcommand names that subcommand there. `port-arg` has to carry the port
in a single argument, because LLDB concatenates the two.

A runtime's diagnostics share the inferior's standard error, so a verbose
runtime breaks tests that match program output. Use its quietest logging.

Most of these values begin with a dash, so the `settings set` examples below
pass `--` first to stop the command from reading them as its own options.

## Example runtimes

### WAMR

[WAMR](https://github.com/bytecodealliance/wasm-micro-runtime)'s `iwasm` takes
the debug server address as `host:port`. Its default heap and stack are small
enough that some tests trap, so raise them.

```
settings set -- platform.plugin.wasm.runtime-path /path/to/iwasm
settings set -- platform.plugin.wasm.runtime-args --heap-size=1048576 --stack-size=1048576 -v=0
settings set -- platform.plugin.wasm.port-arg -g=127.0.0.1:
settings set -- platform.plugin.wasm.env-arg --env=
```

### WasmKit

[WasmKit](https://github.com/swiftwasm/WasmKit)'s CLI takes a bare port number
and defaults to quiet, so `run` is all it needs beyond the port. That subcommand
has to come first, which is what `runtime-args` is for.

```
settings set -- platform.plugin.wasm.runtime-path /path/to/wasmkit-cli
settings set -- platform.plugin.wasm.runtime-args run
settings set -- platform.plugin.wasm.port-arg --debugger-port=
settings set -- platform.plugin.wasm.env-arg --env=
```

:::{note}
WasmKit's debugger support is behind a package trait. The stub is only served by
a CLI built with that trait enabled.
:::

## Running the test suite

Point the test suite at the sysroot and the resource directory, and pass the
target and platform configuration through `LLDB_TEST_USER_ARGS`:

```
$ cmake -G Ninja \
  -DLLDB_TEST_SYSROOT=/path/to/wasi-sdk/share/wasi-sysroot \
  -DLLDB_TEST_RESOURCE_DIR=/path/to/wasi-sdk/lib/clang/<version> \
  -DLLDB_TEST_USER_ARGS="--triple;wasm32-wasip1;--platform-name;wasm;--setting;platform.plugin.wasm.runtime-path=/path/to/iwasm;--setting;platform.plugin.wasm.port-arg=-g=127.0.0.1:;--setting;platform.plugin.wasm.env-arg=--env=;--setting;platform.plugin.wasm.runtime-args=--heap-size=1048576 --stack-size=1048576 -v=0" \
  <other options>
$ ninja check-lldb-api
```

`lldb-dotest` takes the same configuration as command line arguments, which is
easier if you keep the build configured for the host. There `-C`, `--sysroot`
and `--resource-dir` replace the three CMake variables. Also pass a separate
`--build-dir` so Wasm and host inferiors do not share an output directory.

Only the API tests build their inferiors, so this configuration does not affect
`check-lldb-shell` or `check-lldb-unit`.
