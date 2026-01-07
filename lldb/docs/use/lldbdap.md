# Getting started with `lldb-dap`

`lldb-dap` brings the power of `lldb` to any editor or IDE that supports the
[Debug Adapter Protocol (DAP)](https://microsoft.github.io/debug-adapter-protocol/).

## Responsibilities of LLDB, `lldb-dap` and IDE Integrations

Under the hood, the UI-based debugging experience is powered by three separate
components:

- LLDB provides general, IDE-independent debugging features, such as:
  loading binaries / core dumps, interpreting debug info, setting breakpoints,
  pretty-printing variables, etc. The `lldb` binary exposes this functionality
  via a command line interface.
- `lldb-dap` exposes LLDB's functionality via the
  "[Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)",
  i.e. a protocol through which various IDEs (Visual Studio Code, Emacs, vim,
  neovim, ...) can interact with a wide range of debuggers (`lldb-dap` and many
  others).
- An IDE specific extension is used to hook lldb-dap and the IDEs DAP
  implementations together for launching a binary.

Since `lldb-dap` builds on top of LLDB, all of LLDB's extensibility mechanisms
such as [Variable Pretty-Printing](https://lldb.llvm.org/use/variable.html),
[Frame recognizers](https://lldb.llvm.org/use/python-reference.html#writing-lldb-frame-recognizers-in-python)
and [Python Scripting](https://lldb.llvm.org/use/python.html) are available
also in `lldb-dap`.

#### Links to IDE Extensions

- Visual Studio Code -
[LLDB DAP Extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap)
<!-- Add other IDE integrations. -->

## Procuring the `lldb-dap` binary

There are multiple ways to obtain this binary:

- Use the binary provided by your toolchain (for example `xcrun -f lldb-dap` on macOS) or contact your toolchain vendor to include it.
- Download one of the release packages from the [LLVM release page](https://github.com/llvm/llvm-project/releases/). The `LLVM-{version}-{operating_system}.tar.xz` packages contain a prebuilt `lldb-dap` binary or check your systems prefered package manager.
- Build it from source (see [LLDB's build instructions](https://lldb.llvm.org/resources/build.html)).

In some cases, a language specific build of `lldb` / `lldb-dap` may also be
available as part of the languages toolchain. For example the
[swift language](https://www.swift.org/) toolchain includes additional language
integrations in `lldb` and the toolchain builds provider both the `lldb` driver
binary and `lldb-dap` binary.

## Launching a program

To launch an executable for debugging, first define a launch configuration tells
`lldb-dap` how to launch the binary.

A simple launch configuration may look like

```json
{
  "type": "lldb-dap",
  "request": "launch",
  "name": "Debug a.out",
  "program": "a.out"
}
```

See the [Configuration Settings Reference](#configuration-settings-reference)
for more information.

# Supported Features

`lldb-dap` supports the following capabilities:

| Capability                            | Supported               | Description                                                                                                                                                                                                                                                                                                           |
| ------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| supportsConfigurationDoneRequest      | Y                       | The debug adapter supports the `configurationDone` request.                                                                                                                                                                                                                                                           |
| supportsFunctionBreakpoints           | Y                       | The debug adapter supports function breakpoints.                                                                                                                                                                                                                                                                      |
| supportsConditionalBreakpoints        | Y                       | The debug adapter supports conditional breakpoints.                                                                                                                                                                                                                                                                   |
| supportsHitConditionalBreakpoints     | Y                       | The debug adapter supports breakpoints that break execution after a specified number of hits.                                                                                                                                                                                                                         |
| supportsEvaluateForHovers             | Y                       | The debug adapter supports a (side effect free) `evaluate` request for data hovers.                                                                                                                                                                                                                                   |
| exceptionBreakpointFilters            | Y                       | Available exception filter options for the `setExceptionBreakpoints` request.                                                                                                                                                                                                                                         |
| supportsStepBack                      | N                       | The debug adapter supports stepping back via the `stepBack` and `reverseContinue` requests.                                                                                                                                                                                                                           |
| supportsSetVariable                   | N                       | The debug adapter supports setting a variable to a value.                                                                                                                                                                                                                                                             |
| supportsRestartFrame                  | Y                       | The debug adapter supports restarting a frame.                                                                                                                                                                                                                                                                        |
| supportsGotoTargetsRequest            | N                       | The debug adapter supports the `gotoTargets` request.                                                                                                                                                                                                                                                                 |
| supportsStepInTargetsRequest          | Y                       | The debug adapter supports the `stepInTargets` request.                                                                                                                                                                                                                                                               |
| supportsCompletionsRequest            | Y                       | The debug adapter supports the `completions` request.                                                                                                                                                                                                                                                                 |
| completionTriggerCharacters           | `['.', '\s', '\t']`     | The set of characters that should trigger completion in a REPL. If not specified, the UI should assume the `.` character.                                                                                                                                                                                             |
| supportsModulesRequest                | Y                       | The debug adapter supports the `modules` request.                                                                                                                                                                                                                                                                     |
| additionalModuleColumns               | N                       | The set of additional module information exposed by the debug adapter.                                                                                                                                                                                                                                                |
| supportedChecksumAlgorithms           | N                       | Checksum algorithms supported by the debug adapter.                                                                                                                                                                                                                                                                   |
| supportsRestartRequest                | Y (for launch requests) | The debug adapter supports the `restart` request. In this case a client should not implement `restart` by terminating and relaunching the adapter but by calling the `restart` request.                                                                                                                               |
| supportsExceptionOptions              | N                       | The debug adapter supports `exceptionOptions` on the `setExceptionBreakpoints` request.                                                                                                                                                                                                                               |
| supportsValueFormattingOptions        | Y                       | The debug adapter supports a `format` attribute on the `stackTrace`, `variables`, and `evaluate` requests.                                                                                                                                                                                                            |
| supportsExceptionInfoRequest          | Y                       | The debug adapter supports the `exceptionInfo` request.                                                                                                                                                                                                                                                               |
| supportTerminateDebuggee              | Y                       | The debug adapter supports the `terminateDebuggee` attribute on the `disconnect` request.                                                                                                                                                                                                                             |
| supportSuspendDebuggee                | N                       | The debug adapter supports the `suspendDebuggee` attribute on the `disconnect` request.                                                                                                                                                                                                                               |
| supportsDelayedStackTraceLoading      | Y                       | The debug adapter supports the delayed loading of parts of the stack, which requires that both the `startFrame` and `levels` arguments and the `totalFrames` result of the `stackTrace` request are supported.                                                                                                        |
| supportsLoadedSourcesRequest          | N                       | The debug adapter supports the `loadedSources` request.                                                                                                                                                                                                                                                               |
| supportsLogPoints                     | Y                       | The debug adapter supports log points by interpreting the `logMessage` attribute of the `SourceBreakpoint`.                                                                                                                                                                                                           |
| supportsTerminateThreadsRequest       | N                       | The debug adapter supports the `terminateThreads` request.                                                                                                                                                                                                                                                            |
| supportsSetExpression                 | Y                       | The debug adapter supports the `setExpression` request.                                                                                                                                                                                                                                                               |
| supportsTerminateRequest              | N                       | The debug adapter supports the `terminate` request.                                                                                                                                                                                                                                                                   |
| supportsDataBreakpoints               | Y                       | The debug adapter supports data breakpoints.                                                                                                                                                                                                                                                                          |
| supportsReadMemoryRequest             | Y                       | The debug adapter supports the `readMemory` request.                                                                                                                                                                                                                                                                  |
| supportsWriteMemoryRequest            | Y                       | The debug adapter supports the `writeMemory` request.                                                                                                                                                                                                                                                                 |
| supportsDisassembleRequest            | Y                       | The debug adapter supports the `disassemble` request.                                                                                                                                                                                                                                                                 |
| supportsCancelRequest                 | Y                       | The debug adapter supports the `cancel` request.                                                                                                                                                                                                                                                                      |
| supportsBreakpointLocationsRequest    | Y                       | The debug adapter supports the `breakpointLocations` request.                                                                                                                                                                                                                                                         |
| supportsClipboardContext              | N                       | The debug adapter supports the `clipboard` context value in the `evaluate` request.                                                                                                                                                                                                                                   |
| supportsSteppingGranularity           | Y                       | The debug adapter supports stepping granularities (argument `granularity`) for the stepping requests.                                                                                                                                                                                                                 |
| supportsInstructionBreakpoints        | Y                       | The debug adapter supports adding breakpoints based on instruction references.                                                                                                                                                                                                                                        |
| supportsExceptionFilterOptions        | N                       | The debug adapter supports `filterOptions` as an argument on the `setExceptionBreakpoints` request.                                                                                                                                                                                                                   |
| supportsSingleThreadExecutionRequests | N                       | The debug adapter supports the `singleThread` property on the execution requests (`continue`, `next`, `stepIn`, `stepOut`, `reverseContinue`, `stepBack`).                                                                                                                                                            |
| supportsDataBreakpointBytes           | Y                       | The debug adapter supports the `asAddress` and `bytes` fields in the `dataBreakpointInfo` request.                                                                                                                                                                                                                    |
| breakpointModes                       | `[]`                    | Modes of breakpoints supported by the debug adapter, such as 'hardware' or 'software'. If present, the client may allow the user to select a mode and include it in its `setBreakpoints` request. Clients may present the first applicable mode in this array as the 'default' mode in gestures that set breakpoints. |
| supportsANSIStyling                   | Y                       | The debug adapter supports ANSI escape sequences in styling of `OutputEvent.output` and `Variable.value` fields.                                                                                                                                                                                                      |

For more information, see
[Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/).

## Debug Console

The Debug Console allows printing variables / expressions and executing lldb
commands. By default, `lldb-dap` tries to auto-detect whether a provided command
is a variable name / expression whose values will be printed to the Debug
Console or a LLDB command. To side-step this auto-detection and execute a LLDB
command, prefix it with the `commandEscapePrefix`.

The auto-detection mode can be adjusted using the `lldb-dap repl-mode` command
in the Debug Console or by adjusting the `--repl-mode [mode]` argument to
`lldb-dap`. The supported modes are `variable`, `command` and `auto`.

# Configuration Settings Reference

In order for `lldb-dap` to know how to start a debug session a launch or attach
configuration may be specified. Different IDEs may have different mechanisms in
place for configuring the launch configuration.

For Visual Studio Code, see [Visual Studio Code's Debugging User Documentation](https://code.visualstudio.com/docs/debugtest/debugging).

## Common configurations

For both launch and attach configurations, lldb-dap accepts the following
`lldb-dap` specific key/value pairs:

| Parameter                         | Type        | Req |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ----------- | :-: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **name**                          | string      |  Y  | A configuration name that will be displayed in the IDE.                                                                                                                                                                                                                                                                                                                                                                                                    |
| **type**                          | string      |  Y  | Must be "lldb-dap".                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **request**                       | string      |  Y  | Must be "launch" or "attach".                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **program**                       | string      |     | Path to the executable to launch.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **sourcePath**                    | string      |     | Specify a source path to remap \"./\" to allow full paths to be used when setting breakpoints in binaries that have relative source paths.                                                                                                                                                                                                                                                                                                                 |
| **sourceMap**                     | [string[2]] |     | Specify an array of path re-mappings. Each element in the array must be a two element array containing a source and destination pathname. Overrides sourcePath.                                                                                                                                                                                                                                                                                            |
| **debuggerRoot**                  | string      |     | Specify a working directory to use when launching lldb-dap. If the debug information in your executable contains relative paths, this option can be used so that `lldb-dap` can find source files and object files that have relative paths.                                                                                                                                                                                                               |
| **commandEscapePrefix**           | string      |     | The escape prefix to use for executing regular LLDB commands in the Debug Console, instead of printing variables. Defaults to a backtick. If it's an empty string, then all expression in the Debug Console are treated as regular LLDB commands.                                                                                                                                                                                                          |
| **customFrameFormat**             | string      |     | If non-empty, stack frames will have descriptions generated based on the provided format. See https://lldb.llvm.org/use/formatting.html for an explanation on format strings for frames. If the format string contains errors, an error message will be displayed on the Debug Console and the default frame names will be used. This might come with a performance cost because debug information might need to be processed to generate the description. |
| **customThreadFormat**            | string      |     | Same as `customFrameFormat`, but for threads instead of stack frames.                                                                                                                                                                                                                                                                                                                                                                                      |
| **displayExtendedBacktrace**      | bool        |     | Enable language specific extended backtraces.                                                                                                                                                                                                                                                                                                                                                                                                              |
| **enableAutoVariableSummaries**   | bool        |     | Enable auto generated summaries for variables when no summaries exist for a given type. This feature can cause performance delays in large projects when viewing variables.                                                                                                                                                                                                                                                                                |
| **enableSyntheticChildDebugging** | bool        |     | If a variable is displayed using a synthetic children, also display the actual contents of the variable at the end under a [raw] entry. This is useful when creating synthetic child plug-ins as it lets you see the actual contents of the variable.                                                                                                                                                                                                      |
| **initCommands**                  | [string]    |     | LLDB commands executed upon debugger startup prior to creating the LLDB target.                                                                                                                                                                                                                                                                                                                                                                            |
| **preRunCommands**                | [string]    |     | LLDB commands executed just before launching/attaching, after the LLDB target has been created.                                                                                                                                                                                                                                                                                                                                                            |
| **stopCommands**                  | [string]    |     | LLDB commands executed just after each stop.                                                                                                                                                                                                                                                                                                                                                                                                               |
| **exitCommands**                  | [string]    |     | LLDB commands executed when the program exits.                                                                                                                                                                                                                                                                                                                                                                                                             |
| **terminateCommands**             | [string]    |     | LLDB commands executed when the debugging session ends.                                                                                                                                                                                                                                                                                                                                                                                                    |

All commands and command outputs will be sent to the debugger console when they
are executed. Commands can be prefixed with `?` or `!` to modify their behavior:

- Commands prefixed with `?` are quiet on success, i.e. nothing is written to
  stdout if the command succeeds.
- Prefixing a command with `!` enables error checking: If a command prefixed
  with `!` fails, subsequent commands will not be run. This is useful if one of
  the commands depends on another, as it will stop the chain of commands.

## Launch configurations

_NOTE:_ Either `program` or `launchCommands` must be specified.

For JSON configurations of `"type": "launch"`, the JSON configuration can
additionally contain the following key/value pairs:

| Parameter                      | Type                   | Req |                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------ | ---------------------- | :-: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **program**                    | string                 |     | Path to the executable to launch.                                                                                                                                                                                                                                                                                                                                                                                                |
| **args**                       | [string]               |     | An array of command line argument strings to be passed to the program being launched.                                                                                                                                                                                                                                                                                                                                            |
| **cwd**                        | string                 |     | The program working directory.                                                                                                                                                                                                                                                                                                                                                                                                   |
| **env**                        | dictionary or [string] |     | Environment variables to set when launching the program. The string format of each environment variable string is "VAR=VALUE" for environment variables with values or just "VAR" for environment variables with no values.                                                                                                                                                                                                      |
| **stopOnEntry**                | boolean                |     | Whether to stop program immediately after launching.                                                                                                                                                                                                                                                                                                                                                                             |
| **runInTerminal** (deprecated) | boolean                |     | Launch the program inside an integrated terminal in the IDE. Useful for debugging interactive command line programs.                                                                                                                                                                                                                                                                                                             |
| **console**                    | string                 |     | Specify where to launch the program: internal console (`internalConsole`), integrated terminal (`integratedTerminal`) or external terminal (`externalTerminal`). Supported from lldb-dap 21.0 version.                                                                                                                                                                                                                           |
| **stdio**                      | [string]               |     | The stdio property specifies the redirection targets for the debuggee's stdio streams. A null value redirects a stream to the default debug terminal. String can be a path to file, named pipe or TTY device. If less than three values are provided, the list will be padded with the last value. Specifying more than three values will create additional file descriptors (4, 5, etc.). Supported from lldb-dap 22.0 version. |
| **launchCommands**             | [string]               |     | LLDB commands executed to launch the program.                                                                                                                                                                                                                                                                                                                                                                                    |

## Attach configurations

_NOTE:_ Either `pid`, `program`, `coreFile`, `attachCommands`
or`gdb-remote-port` must be specified.

For JSON configurations of `"type": "attach"`, the JSON configuration can
contain the following `lldb-dap` specific key/value pairs:

| Parameter           | Type     | Req |                                                                                                                                                                                                                                                                                                                                              |
| ------------------- | -------- | :-: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **program**         | string   |     | Path to the executable to attach to. This value is optional but can help to resolve breakpoints prior the attaching to the program.                                                                                                                                                                                                          |
| **pid**             | number   |     | The process id of the process you wish to attach to. If **pid** is omitted, the debugger will attempt to attach to the program by finding a process whose file name matches the file name from **program**. Setting this value to `${command:pickMyProcess}` will allow interactive process selection in the Visual Studio Code.             |
| **waitFor**         | boolean  |     | Wait for the process to launch.                                                                                                                                                                                                                                                                                                              |
| **attachCommands**  | [string] |     | LLDB commands that will be executed after **preRunCommands** which take place of the code that normally does the attach. The commands can create a new target and attach or launch it however desired. This allows custom launch and attach configurations. Core files can use `target create --core /path/to/core` to attach to core files. |
| **gdb-remote-port** | int      |     | TCP/IP port to attach to a remote system. Specifying both pid and port is an error.                                                                                                                                                                                                                                                          |
| **gdb-remote-host** | string   |     | The hostname to connect to a remote system. The default hostname being used `localhost`.                                                                                                                                                                                                                                                     |
