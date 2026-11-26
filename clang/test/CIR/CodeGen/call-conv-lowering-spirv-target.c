// Checks which SPIR-V triples run CallConvLowering in the Clang pipeline. The
// SPIR-V ABI extends a char argument with signext, as classic codegen does on
// every triple here. Where the pass runs, ClangIR matches classic codegen
// (LLVM); elsewhere it matches ClangIR with the pass turned off (CIR-NO-CCL).

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -fno-clangir-call-conv-lowering -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-NO-CCL

// RUN: %clang_cc1 -triple spirv32-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-intel -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// AMDGCN-flavored SPIR-V and legacy SPIR don't run the pass.
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-NO-CCL

// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-NO-CCL

// Neither does OpenMP device code.
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-NO-CCL

// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple spirv64-intel -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple spirv64-intel -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=CIR-NO-CCL

#pragma omp declare target
// LLVM: define {{.*}}void @take_char(i8 noundef signext %{{[^,)]+}})
// CIR-NO-CCL: define {{.*}}void @take_char(i8 noundef %{{[^,)]+}})
void take_char(char c) {}
#pragma omp end declare target
