# clang-ssaf-orchestrator-make

Runs a Scalable Static Analysis Framework analysis over a project by riding
along with its `make` build. Each compilation, archive and link step the build
performs is paired with an SSAF step that produces the corresponding summary,
and the summaries are linked into one whole-program view.

- Using it: [Make-Based Projects](
  ../../docs/ScalableStaticAnalysis/user-docs/MakeProjects.md)
- How the shims work: [Make Orchestrator Shims](
  ../../docs/ScalableStaticAnalysis/developer-docs/MakeOrchestratorShims.md)

## Layout

| | |
|---|---|
| `bin/` | the entry script, installed to `bin/` |
| `lib/clang_ssaf_orchestrator_make/` | the package, installed to `lib/` |
| `…/shims/` | the shim bodies and their utilities |
| `shims/` | the shim program, installed to `libexec/ssaf-orchestrator-shims/` |
| `tests/` | the Python unit tests |

The shim program is what a build reaches: at the start of a run the orchestrator
fills `<output-dir>/shims/` with links named after the tools a build might
spell, all pointing at it, and prepends that directory to `PATH`. It is
symlinked under several names, so `argv[0]` is what tells it which tool it
stands in for, and it finds the package by walking up from its own location —
which is why it is a file beside the package rather than part of it.

## Running the tests

```
PYTHONPATH=lib python3 -m unittest discover  # from this directory
llvm-lit clang/test/Analysis/Scalable/ssaf-orchestrator-make
```

The lit suite runs the unit tests too, against the copy in the build tree, so
`ninja check-clang` covers both. The tool is a declared dependency of
`check-clang`.

## Packaging

CMake copies the Python files into the build tree and installs them under the
`clang-ssaf-orchestrator-make` component. The component is not listed in any
distribution cache, so a distribution build will not pick it up without being
told to.

There is no `pyproject.toml`. The entry point is installed by CMake, as it is
for every other Python tool under `clang/tools/`, so a manifest here would
declare an installation path that does not exist. The Python version floor is
the monorepo's, recorded in the root `pyproject.toml`.
