# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Configuration

Use the top-level `build.sh` script for all builds:

```bash
./build.sh debug x86              # → build_debug_x86/ (dev daily driver)
./build.sh release x86            # → build_release_x86/ (for EJIT tests)
./build.sh release x86 minimal    # → build_release_x86_minimal/
./build.sh debug aarch64          # → build_debug_aarch64/
./build.sh release aarch64        # → build_release_aarch64/
```

### Incremental Build (use these daily)

This machine has **limited CPU and memory**. Use targeted incremental builds:

```bash
# Dev build (debug, shared libs)
ninja -C build_debug_x86 clang opt lld

# Release build (static libs, for ejit_test)
ninja -C build_release_x86 LLVMEJIT lld

# Just the AOT passes library (runs inside clang)
ninja -C build_debug_x86 clang

# Just the runtime library (linked into test binaries)
ninja -C build_release_x86 LLVMEJIT
```

### Build directory layout

| Directory | Type | Arch | Purpose |
|-----------|------|------|---------|
| `build_debug_x86/` | Debug | x86 | Dev: clang, opt, lld (shared libs) |
| `build_release_x86/` | Release | x86 | EJIT tests: LLVMEJIT.a, lld |
| `build_debug_aarch64/` | Debug | AArch64 | Cross-compile dev |
| `build_release_aarch64/` | Release | AArch64 | Cross-compile runtime |

## Running Tests

This project uses **lit** (LLVM Integrated Tester) with FileCheck.

- **All LLVM tests**: `cd build && ninja check-llvm`
- **Single lit test file**: `cd build && ./bin/llvm-lit -v ../llvm/test/Transforms/InstCombine/some-test.ll`
- **Directory of lit tests**: `cd build && ./bin/llvm-lit -v ../llvm/test/Transforms/InstCombine/`
- **Specific test subdirectory**: `cd build && ninja check-llvm-transforms-instcombine`
- **Clang lit tests**: `cd build && ninja check-clang`
- **Single clang test**: `cd build && ./bin/llvm-lit -v ../clang/test/CodeGen/some-test.c`
- **LLVM unit tests (gtest)**: `cd build && ninja check-llvm-unit`

### Lit Test Format for LLVM Passes

```llvm
; RUN: opt -passes=<pass-name> -S %s | FileCheck %s
; CHECK: <expected output>
```

### Lit Test Format for Clang (CodeGen / Sema)

```c
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s       // CodeGen
// RUN: %clang_cc1 -fsyntax-only -verify %s                  // Sema diagnostics
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
