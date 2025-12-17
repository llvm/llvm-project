# Getting started with `lldb-dap`

Welcome to the `llb-dap` documentation!

`lldb-dap` brings the power of `lldb` into any editor or IDE that supports the [Debug Adapter Protocol (DAP)](https://microsoft.github.io/debug-adapter-protocol/).

## Prerequisites

In order to begin debugging with `lldb-dap`, you may first need to acquire the
`lldb-dap` binary from an LLVM distribution. For general LLVM releases visit
https://releases.llvm.org/ or check your systems preferred package manager for
the `lldb` package.

In some cases, a language specific build of `lldb` / `lldb-dap` may also be
available as part of the languages toolchain. For example the
[swift language](https://www.swift.org/) toolchain includes additional language integrations in `lldb` and the toolchain builds provider both the `lldb` driver binary and `lldb-dap` binary.

## IDE Integration

In addition to the `lldb-dap` binary, some IDEs have additional extensions to
support debugging.

- Visual Studio Code -
[LLDB DAP Extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap)
<!-- Add other IDE integrations. -->

## Launching a program

To launch an executable for debugging, first define a launch configuration tells `lldb-dap` how to launch the binary.

A simple launch configuration may look like

```json
{
  "type": "lldb-dap",
  "request": "launch",
  "name": "Debug a.out",
  "program": "a.out"
}
```

See the [Configuration Settings Reference](#configuration-settings-reference) for more information.

# Supported Features

`lldb-dap` supports many features of the DAP spec.

* Breakpoints
  * Source breakpoints
  * Function breakpoint
  * Exception breakpoints
* Call Stacks
* Variables
* Watch points
* Expression Evaluation
* And more...

For more information, visit
[Visual Studio Code's Debugging User Documentation](https://code.visualstudio.com/docs/debugtest/debugging)

## Debug Console

The Debug Console allows printing variables / expressions and executing lldb
commands. By default, `lldb-dap` tries to auto-detect whether a provided command
is a variable name / expression whose values will be printed to the Debug
Console or a LLDB command. To side-step this auto-detection and execute a LLDB
command, prefix it with the `commandEscapePrefix`.

The auto-detection mode can ba adjusted using the `lldb-dap repl-mode` command in the Debug Console or by adjusting the `--repl-mode [mode]` argument to `lldb-dap`. The supported modes are `variable`, `command` and `auto`.

# Configuration Settings Reference

## Common configurations

For both launch and attach configurations, lldb-dap accepts the following
`lldb-dap` specific key/value pairs:

| Parameter                         | Type        | Req |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ----------- | :-: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **name**                          | string      |  Y  | A configuration name that will be displayed in the IDE.                                                                                                                                                                                                                                                                                                                                                                                                    |
| **type**                          | string      |  Y  | Must be "lldb-dap".                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **request**                       | string      |  Y  | Must be "launch" or "attach".                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **program**                       | string      |  Y  | Path to the executable to launch.                                                                                                                                                                                                                                                                                                                                                                                                                          |
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

For JSON configurations of `"type": "launch"`, the JSON configuration can
additionally contain the following key/value pairs:

| Parameter                      | Type       | Req |                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------ | ---------- | :-: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **program**                    | string     |  Y  | Path to the executable to launch.                                                                                                                                                                                                                                                                                                                                                                                                |
| **args**                       | [string]   |     | An array of command line argument strings to be passed to the program being launched.                                                                                                                                                                                                                                                                                                                                            |
| **cwd**                        | string     |     | The program working directory.                                                                                                                                                                                                                                                                                                                                                                                                   |
| **env**                        | dictionary |     | Environment variables to set when launching the program. The format of each environment variable string is "VAR=VALUE" for environment variables with values or just "VAR" for environment variables with no values.                                                                                                                                                                                                             |
| **stopOnEntry**                | boolean    |     | Whether to stop program immediately after launching.                                                                                                                                                                                                                                                                                                                                                                             |
| **runInTerminal** (deprecated) | boolean    |     | Launch the program inside an integrated terminal in the IDE. Useful for debugging interactive command line programs.                                                                                                                                                                                                                                                                                                             |
| **console**                    | string     |     | Specify where to launch the program: internal console (`internalConsole`), integrated terminal (`integratedTerminal`) or external terminal (`externalTerminal`). Supported from lldb-dap 21.0 version.                                                                                                                                                                                                                           |
| **stdio**                      | [string]   |     | The stdio property specifies the redirection targets for the debuggee's stdio streams. A null value redirects a stream to the default debug terminal. String can be a path to file, named pipe or TTY device. If less than three values are provided, the list will be padded with the last value. Specifying more than three values will create additional file descriptors (4, 5, etc.). Supported from lldb-dap 22.0 version. |
| **launchCommands**             | [string]   |     | LLDB commands executed to launch the program.                                                                                                                                                                                                                                                                                                                                                                                    |

## Attach configurations

For JSON configurations of `"type": "attach"`, the JSON configuration can
contain the following `lldb-dap` specific key/value pairs:

| Parameter          | Type     | Req |                                                                                                                                                                                                                                                                                                                                              |
| ------------------ | -------- | :-: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **program**        | string   |     | Path to the executable to attach to. This value is optional but can help to resolve breakpoints prior the attaching to the program.                                                                                                                                                                                                          |
| **pid**            | number   |     | The process id of the process you wish to attach to. If **pid** is omitted, the debugger will attempt to attach to the program by finding a process whose file name matches the file name from **program**. Setting this value to `${command:pickMyProcess}` will allow interactive process selection in the IDE.                            |
| **waitFor**        | boolean  |     | Wait for the process to launch.                                                                                                                                                                                                                                                                                                              |
| **attachCommands** | [string] |     | LLDB commands that will be executed after **preRunCommands** which take place of the code that normally does the attach. The commands can create a new target and attach or launch it however desired. This allows custom launch and attach configurations. Core files can use `target create --core /path/to/core` to attach to core files. |
