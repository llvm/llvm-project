# LLVM Project AI Coding Agent Instructions

## Architecture Overview

LLVM is a compiler infrastructure with modular components:
- **Core LLVM** (`llvm/`): IR processing, optimizations, code generation
- **Clang** (`clang/`): C/C++/Objective-C frontend 
- **LLD** (`lld/`): Linker
- **libc++** (`libcxx/`): C++ standard library
- **Target backends** (`llvm/lib/Target/{AMDGPU,X86,ARM,...}/`): Architecture-specific code generation

## Essential Development Workflows

### Build System (CMake + Ninja)
```bash
# Configure with common options for development
cmake -G Ninja -S llvm-project/llvm -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON

# Build and install
cmake --build build
cmake --install build --prefix install/
```

### Testing with LIT
- Use `opt < file.ll -passes=instcombine -S | FileCheck %s` pattern for IR transforms
- Test files go in `llvm/test/Transforms/{PassName}/` with `.ll` extension
- Always include both positive and negative test cases
- Use `CHECK-LABEL:` for function boundaries, `CHECK-NEXT:` for strict sequence

### Key Patterns for Transforms

**InstCombine Pattern** (`llvm/lib/Transforms/InstCombine/`):
- Implement in `InstCombine*.cpp` using visitor pattern (`visitCallInst`, `visitBinaryOperator`)
- Use `PatternMatch.h` matchers: `match(V, m_Add(m_Value(X), m_ConstantInt()))`
- Return `nullptr` for no change, modified instruction, or replacement
- Add to worklist with `Worklist.pushValue()` for dependent values

**Target-Specific Intrinsics**:
- AMDGPU: `@llvm.amdgcn.*` intrinsics in `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`
- Pattern: `if (II->getIntrinsicID() == Intrinsic::amdgcn_ballot)`

## Code Quality Standards

### Control Flow & Debug Info
When modifying control flow, ensure changes don't corrupt:
- Performance profiling data (branch weights, call counts)
- Debug information for branches and calls
- Exception handling unwind information

### Target-Specific Considerations
- **AMDGPU**: Wavefront uniformity analysis affects ballot intrinsics
- **X86**: Vector width and ISA feature dependencies
- Use `TargetTransformInfo` for cost models and capability queries

### Testing Requirements
- Every optimization needs regression tests showing before/after IR
- Include edge cases: constants, undef, poison values
- Test target-specific intrinsics with appropriate triple
- Use `; RUN: opt < %s -passes=... -S | FileCheck %s` format

## Common Development Pitfalls
- Don't assume instruction operand order without checking `isCommutative()`
- Verify type compatibility before creating new instructions
- Consider poison/undef propagation in optimizations
- Check for side effects before eliminating instructions

## Pass Pipeline Context
- InstCombine runs early and multiple times in the pipeline
- Subsequent passes like SimplifyCFG will clean up control flow
- Use `replaceAllUsesWith()` carefully to maintain SSA form
