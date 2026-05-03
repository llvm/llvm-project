# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Configuration

- Build directory: `build/` (out-of-tree CMake build)
- Configured targets: `X86`
- Enabled projects: `clang;lld`
- Build type: `Release`
- Compiler: GCC 13 (`/usr/bin/c++`, `/usr/bin/cc`)
- Build command: `cd build && make -j$(nproc)`
- Specific target: `cd build && make <target-name>` (e.g., `make opt`, `make clang`)

## Running Tests

This project uses **lit** (LLVM Integrated Tester) with FileCheck.

- **All LLVM tests**: `cd build && make check-llvm`
- **Single lit test file**: `cd build && ./bin/llvm-lit ../llvm/test/Transforms/InstCombine/some-test.ll`
- **Directory of lit tests**: `cd build && ./bin/llvm-lit ../llvm/test/Transforms/InstCombine/`
- **Specific test subdirectory**: `cd build && make check-llvm-transforms-instcombine`
- **Check available test targets**: `cd build && ls test/CMakeFiles/ | grep check-llvm`
- **Verbose lit output**: add `-v` to llvm-lit: `./bin/llvm-lit -v ../llvm/test/.../test.ll`
- **LLVM unit tests**: `cd build && make check-llvm-unit`

### Lit Test Format for LLVM Passes

```llvm
; RUN: opt -passes=<pass-name> -S %s | FileCheck %s
; CHECK: <expected output>
```

### Clang tests

```bash
cd build && make check-clang
cd build && ./bin/llvm-lit ../clang/test/CodeGen/some-test.c
```

## Rebuild After Changing Files

- **LLVM libraries/passes**: `cd build && make -j$(nproc) opt` (or just `make` if unsure)
- **Clang changes**: `cd build && make -j$(nproc) clang`
- **Header-only changes**: usually requires `make -j$(nproc)` to rebuild dependents
- **CMake changes**: `cd build && cmake . && make -j$(nproc)`
- **Reconfigure from scratch**:
  ```bash
  cd build
  cmake -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    ../llvm
  ```

## Repository Architecture

This is the **LLVM monorepo**. Key top-level directories:

| Directory | Purpose |
|-----------|---------|
| `llvm/` | Core LLVM: IR, passes, codegen, execution engines |
| `clang/` | C/C++/ObjC frontend |
| `lld/` | LLVM linker |
| `compiler-rt/` | Compiler runtime (sanitizers, builtins) |
| `libcxx/`, `libcxxabi/`, `libunwind/` | C++ standard library |
| `mlir/` | MLIR framework |
| `bolt/` | Binary optimization tool |
| `flang/` | Fortran frontend |

### LLVM Passes

- **Pass implementations**: `llvm/lib/Transforms/<PassCategory>/`
- **Pass registration** (new PM): `llvm/lib/Passes/PassRegistry.def` — each pass listed as `MODULE_PASS("name", ClassName())` or `FUNCTION_PASS("name", ClassName())`
- **PassBuilder**: `llvm/lib/Passes/PassBuilder.cpp` — pipeline construction and analysis registration
- Passes use `PassInfoMixin<ClassName>` pattern (new pass manager); no manual registration needed beyond `PassRegistry.def`

### LLVM Execution Engine / OrcJIT

- `llvm/lib/ExecutionEngine/` — JIT infrastructure: OrcJIT, JITLink, Interpreter, MCJIT
- A new EmbeddedJIT runtime library is planned at `llvm/lib/ExecutionEngine/EJIT/` (not yet created on this branch)
- Corresponding headers go in `llvm/include/llvm/ExecutionEngine/EJIT/`

### Clang Architecture

- **Attribute definitions**: `clang/include/clang/Basic/Attr.td` (TableGen, ~5200 lines)
- **Semantic analysis**: `clang/lib/Sema/`
- **CodeGen (AST → LLVM IR)**: `clang/lib/CodeGen/`
- **Backend integration** (adding LLVM passes to Clang pipeline): `clang/lib/CodeGen/BackendUtil.cpp`

### Testing

- **Lit tests** (LLVM IR / opt / llc): `llvm/test/<category>/<test>.ll`
- **Unit tests** (C++ / gtest): `llvm/unittests/<category>/<test>.cpp`
- **Clang lit tests**: `clang/test/<category>/<test>.c` or `.cpp`
- **Execution Engine integration tests**: `llvm/test/ExecutionEngine/`

## This Branch: `ejit_dev`

This branch is developing **EmbeddedJIT** — an embedded-scenario JIT compilation system based on time-window constants with runtime specialization. Design documents are in `/workspaces/jit_design_doc/`.

### Planned file locations (to be created):

- **AOT Passes**: `llvm/lib/Transforms/EmbeddedJIT/` — 5 passes (EJitRegisterBitcode, EJitRegisterPeriod, EJitWrapperGen, EJitPeriodHandler, EJitAotModulePass)
- **Runtime library**: `llvm/lib/ExecutionEngine/EJIT/` — core engine, cache, compiler, optimizer, PASS6
- **Runtime headers**: `llvm/include/llvm/ExecutionEngine/EJIT/`
- **AOT Pass lit tests**: `llvm/test/Transforms/EmbeddedJIT/`
- **Runtime unit tests**: `llvm/unittests/ExecutionEngine/EJIT/`
- **Clang attributes**: `clang/include/clang/Basic/Attr.td`, `clang/lib/Sema/`, `clang/lib/CodeGen/`

### Key design decisions (from SPEC4.md / PLAN4.md):

- **Single-function wrapper**: wrapper logic is inserted directly into the original `ejit_entry` function entry, no separate wrapper function
- **Metadata-driven**: `!ejit.may_const` on load instructions (soft annotation, safe to drop), `!ejit.metadata` on functions/globals
- **Two-phase AOT**: early pass (before O2/O3) extracts raw bitcode with metadata intact; late passes (after O2/O3) add wrapper + period registration
- **OrcJIT + JITLink**: runtime uses LLJIT with custom embedded memory manager (slab allocator, 512KB default)
- **JIT pipeline order**: param substitution → InstCombine → Inline → EJitStructFieldPass → standard LLVM opts (L1/L2/L3)

## Code Review Considerations

When reviewing changes that modify function control flow, pay close attention to whether the change could corrupt performance profile data or invalidate debug information, particularly for branches and calls. (From `.github/copilot-instructions.md`)
