# vscode-clangd

Provides C/C++ language IDE features for VS Code using [clangd](https://clang.llvm.org/extra/clangd.html).

## Usage

`vscode-clangd` provides the features designated by the [Language Server
Protocol](https://github.com/Microsoft/language-server-protocol), such as
code completion, code formatting and goto definition.

**Note**: `clangd` is under heavy development, not all LSP features are
implemented. See [Current Status](https://clang.llvm.org/extra/clangd.html#current-status)
for details.

To use `vscode-clangd` extension in VS Code, you need to install `vscode-clangd`
from VS Code extension marketplace.

`vscode-clangd` will attempt to find the `clangd` binary on your `PATH`.
Alternatively, the `clangd` executable can be specified in your VS Code
`settings.json` file:

```json
{
    "clangd.path": "/absolute/path/to/clangd"
}
```

To obtain `clangd` binary, please see the [installing Clangd](https://clang.llvm.org/extra/clangd.html#installing-clangd).

## Development

A guide of developing `vscode-clangd` extension.

### Requirements

* VS Code
* node.js and npm

### Steps

1. Make sure you disable the installed `vscode-clangd` extension in VS Code.
2. Make sure you have clangd in /usr/bin/clangd or edit src/extension.ts to
point to the binary.
3. In order to start a development instance of VS code extended with this, run:

```bash
   $ cd /path/to/clang-tools-extra/clangd/clients/clangd-vscode/
   $ npm install
   $ code .
   # When VS Code starts, press <F5>.
```
