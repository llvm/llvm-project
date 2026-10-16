// A missing fat binary must produce the same clang diagnostic from ClangIR as
// from classic codegen, not an MLIR pass error.

// RUN: not %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.nonexistent \
// RUN:   -o %t.cir 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.nonexistent \
// RUN:   -o %t.ll 2>&1 | FileCheck %s

// RUN: not %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.nonexistent \
// RUN:   -o %t-ogcg.ll 2>&1 | FileCheck %s

#include "Inputs/cuda.h"

// A kernel is needed: with nothing to register neither CIRGen nor classic
// codegen reads the fat binary at all.
__global__ void kernel() {}

// CHECK: fatal error: cannot open file '{{.*}}.nonexistent':
