# Contributing to LLDB-DAP

This guide describes how to extend and contribute to lldb-dap.
For documentation on how to use lldb-dap, see [lldb-dap's README](https://github.com/llvm/llvm-project/blob/main/lldb/tools/lldb-dap/README.md).

lldb-dap and LLDB are developed under the umbrella of the
[LLVM project](https://llvm.org/). As such, the
"[Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html)",
"[Contributing to LLVM](https://llvm.org/docs/Contributing.html)" and
"[LLVM coding standard](https://llvm.org/docs/CodingStandards.html)"
guides might also be relevant, if you are a first-time contributor to the LLVM
project.

lldb-dap's source code is [part of the LLVM
repository](https://github.com/llvm/llvm-project/tree/main/lldb/tools/lldb-dap)
on Github. We use Github's [issue
tracker](https://github.com/llvm/llvm-project/tree/main/lldb/tools/lldb-dap)
and patches can be submitted via [pull
requests](https://github.com/llvm/llvm-project/pulls).

## Building `lldb-dap` from soruce

To build lldb-dap from source, first need to [setup a LLDB build](https://lldb.llvm.org/resources/build.html).
After doing so, run `ninja lldb-dap`. To use your freshly built `lldb-dap`
binary, install the VS Code extension and point it to lldb-dap by setting the
`lldb-dap.executable-path` setting.

## Responsibilities of LLDB, lldb-dap and the Visual Studio Code Extension

Under the hood, the UI-based debugging experience is fueled by three separate
components:

* LLDB provides general, IDE-indepedent debugging features, such as:
  loading binaries / core dumps, interpreting debug info, setting breakpoints,
  pretty-printing variables, etc. The `lldb` binary exposes this functionality
  via a command line interface.
* `lldb-dap` exposes LLDB's functionality via the
  "[Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)",
  i.e. a protocol through which various IDEs (VS Code, Emacs, vim, neovim, ...)
  can interact with a wide range of debuggers (`lldb-dap` and many others).
* The VS Code extension exposes the lldb-dap binary within VS Code. It acts
  as a thin wrapper around the lldb-dap binary, and adds VS-Code-specific UI
  integration on top of lldb-dap, such as autocompletion for `launch.json`
  configuration files.

Since lldb-dap builds on top of LLDB, all of LLDB's extensibility mechanisms
such as [Variable Pretty-Printing](https://lldb.llvm.org/use/variable.html),
[Frame recognizers](https://lldb.llvm.org/use/python-reference.html#writing-lldb-frame-recognizers-in-python)
and [Python Scripting](https://lldb.llvm.org/use/python.html) are available
also in lldb-dap.

When adding new functionality, you generally want to add it on the lowest
applicable level. I.e., quite frequently you actually want to add functionality
to LLDB's core in order to improve your debugging experience in VS Code.

### The Debug Adapter Protocol

The most relevant resources for the Debug Adapter Protocol are:
* [The overview](https://microsoft.github.io/debug-adapter-protocol/overview)
  which provides a high-level introduction,
* the [human-readable specification](https://microsoft.github.io/debug-adapter-protocol/specification), and
* the [JSON-schema specification](https://github.com/microsoft/debug-adapter-protocol/blob/main/debugAdapterProtocol.json).

lldb-dap adds some additional non-standard extensions to the protocol. To take
advantage of those extensions, IDE-specific support code is needed, usually
inside the VS Code extension. When adding a new extension, please first look
through the [issue tracker of the Debug Adapter
Protocol](https://github.com/microsoft/debug-adapter-protocol/issues) to check
if there already is a proposal serving your use case. If so, try to take
inspiration from it. If not, consider opening an upstream issue.

To avoid naming collisions with potential future extensions of the Debug
Adapter protocol, all non-standard extensions should use the prefix
`$__lldb_extension` in their JSON property names.

### Debugging the Debug Adapter Protocol

To debug the Debug Adapter Protocol, point the `LLDBDAP_LOG` environment
variable to a file on your disk. lldb-dap will log all communication received
from / sent to the IDE to the provided path. In the VS Code extension, you
can also set the log path through the `lldb-dap.log-path` VS Code setting.

## Building the VS Code extension from source

Installing the plug-in is very straightforward and involves just a few steps.

In most cases, installing the VS Code extension from the [VS Code
Marketplace](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap)
and pointing it to a locally built lldb-dap binary is sufficient. Building
the VS Code extension from source is only necessary if the TypeScript code is
changed.

### Pre-requisites

- Install a modern version of node (e.g. `v20.0.0`).
- On VS Code, execute the command `Install 'code' command in PATH`. You need to
  do it only once. This enables the command `code` in the PATH.

### Packaging and installation

```bash
cd /path/to/lldb/tools/lldb-dap
npm install
npm run package # This also compiles the extension.
npm run vscode-install
```

On VS Code, set the setting `lldb-dap.executable-path` to the path of your local
build of `lldb-dap`.

And then you are ready!

### Updating the extension

Updating the extension is pretty much the same process as installing it from
scratch. However, VS Code expects the version number of the upgraded extension
to be greater than the previous one, otherwise the installation step might have
no effect.

```bash
# Bump version in package.json
cd /path/to/lldb/tools/lldb-dap
npm install
npm run package
npm run vscode-install
```

Another way upgrade without bumping the extension version is to first uninstall
the extension, then reload VS Code, and then install it again. This is
an unfortunate limitation of the editor.

```bash
cd /path/to/lldb/tools/lldb-dap
npm run vscode-uninstall
# Then reload VS Code: reopen the IDE or execute the `Developer: Reload Window`
# command.
npm run package
npm run vscode-install
```

### Deploying for Visual Studio Code

The easiest way to deploy the extension for execution on other machines requires
copying `lldb-dap` and its dependencies into a`./bin` subfolder and then create a
standalone VSIX package.

```bash
cd /path/to/lldb/tools/lldb-dap
mkdir -p ./bin
cp /path/to/a/built/lldb-dap ./bin/
cp /path/to/a/built/liblldb.so ./bin/
npm run package
```

This will produce the file `./out/lldb-dap.vsix` that can be distributed. In
this type of installation, users don't need to manually set the path to
`lldb-dap`. The extension will automatically look for it in the `./bin`
subfolder.

*Note: It's not possible to use symlinks to `lldb-dap`, as the packaging tool
forcefully performs a deep copy of all symlinks.*

*Note: It's possible to use this kind flow for local installations, but it's
not recommended because updating `lldb-dap` requires rebuilding the extension.*

## Formatting the Typescript code

This is also very simple, just run:

```bash
npm run format
```

## Working with the VS Code extension from another extension

The VS Code extension exposes the following [VS Code
commands](https://code.visualstudio.com/api/extension-guides/command),
which can be invoked by other debugger extensions to leverage this extension's
settings and logic. The commands help resolve configuration, create adapter
descriptor, and get the lldb-dap process for state tracking, additional
interaction, and telemetry.

```
// Resolve debug configuration
const resolvedConfiguration = await vscode.commands.executeCommand("lldb-dap.resolveDebugConfiguration", folder, configuration, token);

// Resolve debug configuration with substituted variables
const resolvedConfigurationWithSubstitutedVariables = await vscode.commands.executeCommand("lldb-dap.resolveDebugConfigurationWithSubstitutedVariables", folder, configuration, token);

// Create debug adapter descriptor
const adapterDescriptor = await vscode.commands.executeCommand("lldb-dap.createDebugAdapterDescriptor", session, executable);

// Get DAP server process
const process = await vscode.commands.executeCommand("lldb-dap.getServerProcess");
```
