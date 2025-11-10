// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-unknown-elf -mavx2 -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang --target=x86_64-unknown-elf -mavx2 -c -O2 -fenable-ripple -fripple-lib=%t.rlib.bc -emit-llvm -S -o - -mllvm -ripple-pad-to-target-simd -mllvm -ripple-disable-link %s | FileCheck %s

#include <ripple.h>

extern float doublify(float);

void foo(const float A[37], float EightA[37]) {
  ripple_block_t BS = ripple_set_block_shape(0, 37);
  size_t v0 = ripple_id(BS, 0);
  EightA[v0] = doublify(4 * A[v0]);
}
// CHECK-LABEL: foo
// CHECK: call <40 x float> @llvm.masked.load.v40f32.p0
// CHECK: fmul <40 x float>
// CHECK: call <32 x float> @ripple_ew_doublify(ptr nonnull
// CHECK: call <32 x float> @ripple_ew_doublify(ptr nonnull
// CHECK: call void @llvm.masked.store.v40f32.p0
// CHECK: ret void
