# Development

A guide of developing `vscode-clangd` extension.

## Requirements

* VS Code
* node.js and npm

## Steps

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

# Contributing

Please follow the exsiting code style when contributing to the extension, we
recommend to run `npm run format` before sending a patch.

# Publish to VS Code Marketplace

New changes to `clangd-vscode` are not released until a new version is published
to the marketplace.

## Requirements

* Make sure install the `vsce` command (`npm install -g vsce`)
* `llvm-vs-code-extensions` account
* Bump the version in `package.json`, and commit the change to upstream

The extension is published under `llvm-vs-code-extensions` account, which is
currently maintained by clangd developers. If you want to make a new release,
please contact clangd-dev@lists.llvm.org.

## Steps

```bash
  $ cd /path/to/clang-tools-extra/clangd/clients/clangd-vscode/
  # For the first time, you need to login in the account. vsce will ask you for
    the Personal Access Token, and remember it for future commands.
  $ vsce login llvm-vs-code-extensions
  # Publish the extension to the VSCode marketplace.
  $ npm run publish
```
