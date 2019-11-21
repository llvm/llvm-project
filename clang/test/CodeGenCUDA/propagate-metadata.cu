// Check that when we link a bitcode module into a file using
// -mlink-builtin-bitcode, we apply the same attributes to the functions in that
// bitcode module as we apply to functions we generate.
//
// In particular, we check that ftz and unsafe-math are propagated into the
// bitcode library as appropriate.
//
// In addition, we set -ftrapping-math on the bitcode library, but then set
// -fno-trapping-math on the main compilations, and ensure that the latter flag
// overrides the flag on the bitcode library.

// Build the bitcode library.  This is not built in CUDA mode, otherwise it
// might have incompatible attributes.  This mirrors how libdevice is built.
// RUN: %clang_cc1 -x c++ -fconvergent-functions -emit-llvm-bc -ftrapping-math -DLIB \
// RUN:   %s -o %t.bc -triple nvptx-unknown-unknown

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc -o - \
// RUN:   -fno-trapping-math -fcuda-is-device -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK --check-prefix=NOFTZ --check-prefix=NOFAST

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc \
// RUN:   -fno-trapping-math -fcuda-flush-denormals-to-zero -o - \
// RUN:   -fcuda-is-device -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK --check-prefix=FTZ \
// RUN:   --check-prefix=NOFAST

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc \
// RUN:   -fno-trapping-math -fcuda-flush-denormals-to-zero -o - \
// RUN:   -fcuda-is-device -menable-unsafe-fp-math -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK --check-prefix=FAST

// Wrap everything in extern "C" so we don't have to worry about name mangling
// in the IR.
extern "C" {
#ifdef LIB

// This function is defined in the library and only declared in the main
// compilation.
void lib_fn() {}

#else

#include "Inputs/cuda.h"
__device__ void lib_fn();
__global__ void kernel() { lib_fn(); }

#endif
}

// The kernel and lib function should have the same attributes.
// CHECK: define void @kernel() [[attr:#[0-9]+]]
// CHECK: define internal void @lib_fn() [[attr]]

// Check the attribute list.
// CHECK: attributes [[attr]] = {
// CHECK-SAME: convergent
// CHECK-SAME: "no-trapping-math"="true"

// FTZ-SAME: "nvptx-f32ftz"="true"
// NOFTZ-NOT: "nvptx-f32ftz"="true"

// FAST-SAME: "unsafe-fp-math"="true"
// NOFAST-NOT: "unsafe-fp-math"="true"
