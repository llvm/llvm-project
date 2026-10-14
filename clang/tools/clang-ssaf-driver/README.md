# clang-ssaf-driver

Runs a Scalable Static Analysis Framework source transformation over a project
as it builds, and presents the result: the source edits as a patch, the
transformation report as diagnostics and as SARIF, and optionally the edits
applied with `clang-apply-replacements`. Projects built with make are driven
through `clang-ssaf-orchestrator-make`, installed beside it.

- Using it: [Transforming a Project](
  ../../docs/ScalableStaticAnalysis/user-docs/Driver.md)

## Layout

| | |
|---|---|
| `bin/` | the entry script, installed to `bin/` |
| `lib/clang_ssaf_driver/` | the package, installed to `lib/` |
| `tests/` | the Python unit tests |

## Running the tests

```
PYTHONPATH=lib python3 -m unittest discover  # from this directory
llvm-lit clang/test/Analysis/Scalable/ssaf-driver
```

The lit suite runs the unit tests too, against the copy in the build tree, so
`ninja check-clang` covers both. The tool is a declared dependency of
`check-clang`.

## Packaging

CMake copies the Python files into the build tree and installs them under the
`clang-ssaf-driver` component, and installing it installs the orchestrator's
component too.
