# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Configuration

- **Build directory**: `build/` (out-of-tree CMake, **Ninja** generator)
- **Build type**: `Debug` (shared libs, split DWARF — optimized for incremental dev)
- **Compiler**: Clang + ccache (`clang`/`clang++` with `ccache` launcher)
- **Linker**: `lld`
- **Targets**: `X86`
- **Enabled projects**: `clang`

### First-time Configure

```bash
cmake -S llvm -B build \
  -G "Ninja" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### Incremental Build (use these daily)

This machine has **limited CPU and memory**. After the first full build, always use targeted incremental builds to avoid OOM and excessive compile time:

```bash
# Build just the `opt` tool (and its library dependencies) — fastest for LLVM pass dev
cd build && ninja opt

# Build just `clang` (for frontend attribute / CodeGen work)
cd build && ninja clang

# Build both opt and clang
cd build && ninja opt clang

# Build specific LLVM library (for runtime lib work)
cd build && ninja LLVMEJIT        # once the EJIT runtime library exists
cd build && ninja LLVMCore        # Core changes
cd build && ninja LLVMTransformUtils

# Rebuild only what changed (Ninja auto-detects, no need for -j flag)
cd build && ninja

# If you must do a full build, limit parallelism to avoid OOM:
cd build && ninja -j2            # 2 parallel jobs max on low-memory machines
```

### Reconfigure After CMake Changes

```bash
cd build && cmake . && ninja      # cmake re-reads cache, ninja rebuilds
```

### ccache

- ccache caches compiled object files across builds. After a `git checkout` or branch switch, most objects hit the cache.
- Check stats: `ccache -s`
- Clear cache: `ccache -C`

### Why Each Config Option

| Option | Reason |
|--------|--------|
| `Debug` | Needed for step-through debugging of clang/opt |
| `BUILD_SHARED_LIBS=ON` | Dramatically smaller link times and memory per binary |
| `LLVM_OPTIMIZED_TABLEGEN=ON` | TableGen runs much faster even in Debug |
| `LLVM_USE_LINKER=lld` | 2-4x faster linking than GNU ld / gold |
| `LLVM_USE_SPLIT_DWARF=ON` | Smaller object files, less I/O during linking |
| `CMAKE_C_COMPILER_LAUNCHER=ccache` | Skip recompilation of unchanged TUs |

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
