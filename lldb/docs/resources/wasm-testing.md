# Debugging and Testing LLDB with WebAssembly

LLDB debugs WebAssembly through a standalone Wasm runtime. The runtime executes
the module and serves a GDB remote stub. LLDB's `wasm` platform launches the
runtime, picks a free TCP port and connects to the stub on it. The LLDB API test
suite uses the same setup to compile its test programs to WebAssembly and run
them.

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

Both compilers need the WASI sysroot, passed as `LLDB_TEST_SYSROOT`.

## Configuring the platform

The `wasm` platform is selected with `--platform-name wasm`. The global
[`platform.plugin.wasm`](/use/settings.md#wasm) settings `runtime-path`,
`port-arg`, `env-arg` and `runtime-args` describe how to invoke the runtime.
`env-arg` is optional. Set it to forward the launch environment
(`target.env-vars`, `process launch -E`).

LLDB assembles the runtime's command line as:

```
<runtime-path> <runtime-args...> <port-arg><port> <env-arg><key=value>... <module> <inferior-args...>
```

`runtime-args` precedes the port argument, so a runtime that dispatches on a
leading subcommand names that subcommand there. `port-arg` has to carry the port
in a single argument, because LLDB concatenates the two. `env-arg` is repeated
once per environment variable, for example `--env=A=1 --env=B=2`.

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
WasmKit's debugger support is behind the `WasmDebuggingSupport` package trait.
A CLI built without it rejects `--debugger-port` as an unknown option. Build
with
`swift build -c release --product wasmkit-cli --traits WasmDebuggingSupport`,
adding whichever other traits you need.
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

The triple, platform and runtime settings reach only the API tests, so
`check-lldb-unit` is unaffected and `check-lldb-shell` still builds for the
host. `LLDB_TEST_SYSROOT` is the exception, since the shell tests also read it
as their host sysroot. Keep the Wasm sysroot out of a build that runs them.
