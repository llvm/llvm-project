// RUN: %clang_cc1 -fcuda-is-device \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=NO-PREC-SQRT %s

// RUN: %clang_cc1 -fcuda-is-device -fcuda-prec-sqrt \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=PREC-SQRT %s

#include "Inputs/cuda.h"

extern "C" __device__ void foo() {}


// NO-PREC-SQRT: !{i32 4, !"nvvm-reflect-prec-sqrt", i32 0}
// PREC-SQRT: !{i32 4, !"nvvm-reflect-prec-sqrt", i32 1}
