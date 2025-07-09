// REQUIRES: hexagon-registered-target
// RUN: %clang -S --target=hexagon -mhvx -mv81 -mhvx-length=128B -O2 -fenable-ripple -fdisable-ripple-lib -mllvm -ripple-pad-to-target-simd -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>

void foo(size_t N, const float a[N], const float b[N], float apb[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 3);
  size_t v0 = ripple_id(BS, 0);
  apb[v0] = a[v0] + b[v0];
}
// CHECK: foo
// CHECK: llvm.masked.load.v32f32
// CHECK: llvm.masked.load.v32f32
// CHECK: llvm.masked.store.v32f32
// CHECK: ret void

void foo_0(size_t N, const float a[N], const float b[N], float apb[N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 12);
  size_t v0 = ripple_id(BS, 0);
  apb[v0] = a[v0] + b[v0];
}
// CHECK: foo_0
// CHECK: llvm.masked.load.v32f32
// CHECK: llvm.masked.load.v32f32
// CHECK: llvm.masked.store.v32f32
// CHECK: ret void

void foo_1(const float x[33], float y[33]) {
  ripple_block_t BS = ripple_set_block_shape(0, 33);
  size_t v0 = ripple_id(BS, 0);
  y[v0] = sinf(x[v0]);
}
// CHECK: foo_1
// CHECK: llvm.masked.load.v64f32
// CHECK: llvm.sin.v64f32
// CHECK: llvm.masked.store.v64f32
// CHECK: ret void

extern float bar(float);
void foo_2(const float x[12], float y[12]) {
  ripple_block_t BS = ripple_set_block_shape(0, 12);
  size_t v0 = ripple_id(BS, 0);
  y[v0] = bar(x[v0]);
}
// CHECK: foo_2
// CHECK: llvm.masked.load.v32f32
// CHECK: call float @bar(float %{{.+}})
// CHECK: llvm.masked.store.v32f32
// CHECK: ret void
