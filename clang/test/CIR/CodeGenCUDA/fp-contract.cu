// CUDA device compilation defaults to -ffp-contract=fast.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#include "Inputs/cuda.h"

__device__ float axpy(float a, float x, float y) { return a * x + y; }

// CIR-LABEL: cir.func {{.*}}@_Z4axpyfff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}

// LLVM-LABEL: @_Z4axpyfff
// LLVM: fmul contract float
// LLVM: fadd contract float
