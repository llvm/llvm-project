// CUDA's default contract mode is -ffp-contract=fast. CIR records that as
// `contract` on fmul/fadd, not as cir.fmuladd. The lowered LLVM IR must carry
// the same flag so a later Standard-fusion backend still emits FMA.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-FAST
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-FAST

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -ffp-contract=on -fclangir -emit-cir %s -o %t-on.cir
// RUN: FileCheck --input-file=%t-on.cir %s -check-prefix=CIR-ON

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --input-file=%t-og.ll %s -check-prefix=LLVM-FAST

#include "Inputs/cuda.h"

__host__ __device__ float same_stmt(float a, float b, float c) {
  return a * b + c;
}
// CIR-FAST-LABEL: cir.func {{.*}}@_Z9same_stmtfff
// CIR-FAST: cir.fmul {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// CIR-ON-LABEL: cir.func {{.*}}@_Z9same_stmtfff
// CIR-ON: cir.fmuladd
// CIR-ON-NOT: #cir.fastmath

// LLVM-FAST-LABEL: @_Z9same_stmtfff
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float
// LLVM-FAST-NOT: @llvm.fmuladd

__host__ __device__ float across_stmt(float a, float b, float c) {
  float t = a * b;
  return t + c;
}
// CIR-FAST-LABEL: cir.func {{.*}}@_Z11across_stmtfff
// CIR-FAST: cir.fmul {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd {{.*}} : !cir.float {fastmath = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// CIR-ON-LABEL: cir.func {{.*}}@_Z11across_stmtfff
// CIR-ON: cir.fmul {{.*}} : !cir.float
// CIR-ON: cir.fadd {{.*}} : !cir.float
// CIR-ON-NOT: cir.fmuladd
// CIR-ON-NOT: #cir.fastmath
// CIR-ON: cir.return

// LLVM-FAST-LABEL: @_Z11across_stmtfff
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float
