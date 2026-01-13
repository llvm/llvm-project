// Check that when we link a bitcode module into a file using
// -mlink-builtin-bitcode, we apply the same attributes to the functions in that
// bitcode module as we apply to functions we generate.
//
// In particular, we check that ftz and unsafe-math are propagated into the
// bitcode library as appropriate.

// Build the bitcode library.  This is not built in CUDA mode, otherwise it
// might have incompatible attributes.  This mirrors how libdevice is built.
// RUN: %clang_cc1 -x c++ -fconvergent-functions -emit-llvm-bc -DLIB \
// RUN:   %s -o %t.bc -triple nvptx-unknown-unknown

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc -o - \
// RUN:   -fcuda-is-device -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK --check-prefix=NOFTZ

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc \
// RUN:   -fdenormal-fp-math-f32=preserve-sign -o - \
// RUN:   -fcuda-is-device -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK --check-prefix=FTZ

// RUN: %clang_cc1 -x cuda %s -emit-llvm -mlink-builtin-bitcode %t.bc \
// RUN:   -fdenormal-fp-math-f32=preserve-sign -o - \
// RUN:   -fcuda-is-device -funsafe-math-optimizations -triple nvptx-unknown-unknown \
// RUN: | FileCheck %s --check-prefix=CHECK

#ifndef LIB
#include "Inputs/cuda.h"
#endif

// Wrap everything in extern "C" so we don't have to worry about name mangling
// in the IR.
extern "C" {
#ifdef LIB

// This function is defined in the library and only declared in the main
// compilation.
void lib_fn() {}

#else

__device__ void lib_fn();
__global__ void kernel() { lib_fn(); }

#endif
}

// The kernel and lib function should have the same attributes.
// CHECK: define{{.*}} void @kernel() [[kattr:#[0-9]+]]
// CHECK: define internal void @lib_fn() [[fattr:#[0-9]+]]

// FIXME: These -NOT checks do not work as intended and do not check on the same
// line.

// Check the attribute list for kernel.
// CHECK: attributes [[kattr]] = {

// CHECK-SAME: convergent
// CHECK-SAME: norecurse

// FTZ-NOT: "denormal-fp-math"
// FTZ-SAME: "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// NOFTZ-NOT: "denormal-fp-math-f32"

// CHECK-SAME: "no-trapping-math"="true"

// Check the attribute list for lib_fn.
// CHECK: attributes [[fattr]] = {

// CHECK-SAME: convergent
// CHECK-NOT: norecurse

// FTZ-NOT: "denormal-fp-math"
// NOFTZ-NOT: "denormal-fp-math"

// FTZ-SAME: "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// NOFTZ-NOT: "denormal-fp-math-f32"

// CHECK-SAME: "no-trapping-math"="true"
