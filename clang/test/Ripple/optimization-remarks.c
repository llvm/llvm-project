// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang -g -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -fenable-ripple -fripple-lib=%t.ll -S -emit-llvm -O2 -ffast-math %s -o %t -Rpass=ripple -mllvm -ripple-disable-link 2> %t.err && FileCheck %s --input-file=%t.err --check-prefix=SUCCESS
// RUN: %clang -fenable-ripple -fripple-lib=%t.ll -S -emit-llvm -O2 -ffast-math %s -o %t -Rpass-missed=ripple -mllvm -ripple-disable-link 2> %t.err && FileCheck %s --input-file=%t.err --check-prefix=FAILURE
#ifdef COMPILE_LIB

#include <stdint.h>

typedef float f32t4 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));

extern inline f32t4 ripple_externf_exact(f32t4 A) { return (A / 2.f) + 42.f; }
extern inline f32t4 ripple_ew_pure_externf_ew(f32t4 A, f32t4 B) {
  return A + B / 2.f + 1.f;
}

#else

#include <ripple.h>

float externf_ew(float, float);
float externf_exact(float);
__attribute__((noinline)) float promotable(float x, float y) {
  return x * x + y;
}

void f(float *a, float *b, float *c, int *d) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  for (int i = 0; i < 4001; i += ripple_get_block_size(BS, 0))
    c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
  if (ripple_id(BS, 0) < 4) {
    float cosVal = __builtin_ripple_cosf(c[ripple_id(BS, 0)]);
    c[ripple_id(BS, 0)] +=
        __builtin_abs(ripple_add_sat((int)cosVal, (int)cosVal));
  }
  c[ripple_id(BS, 0)] += externf_ew(
      promotable(externf_exact(c[ripple_id(BS, 0)]), a[ripple_id(BS, 0)]), 1.f);

  c[ripple_id(BS, 0)] += promotable(c[ripple_id(BS, 0)], 42.f);

  ripple_block_t BS2 = ripple_set_block_shape(0, 4);
  c[ripple_id(BS2, 0)] +=
      externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);

  c[ripple_id(BS2, 0)] += ripple_reduceadd(0x1, c[ripple_id(BS2, 0)] + 32.f);

  ripple_block_t BS3 = ripple_set_block_shape(0, 1, 4);
  c[ripple_id(BS, 0) + ripple_id(BS3, 1)] =
      a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
}

#endif

// SUCCESS:      optimization-remarks.c:23:1: remark: function 'promotable' specialized for tensor operands {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    23 | __attribute__((noinline)) float promotable(float x, float y) {
// SUCCESS-NEXT:       | ^
// SUCCESS-NEXT: optimization-remarks.c:24:3: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |   ^
// SUCCESS-NEXT: optimization-remarks.c:24:12: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |            ^
// SUCCESS-NEXT: optimization-remarks.c:24:16: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |                ^
// SUCCESS-NEXT: optimization-remarks.c:23:1: remark: function 'promotable' specialized for tensor operands {shape: (Tensor[32], unchanged) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    23 | __attribute__((noinline)) float promotable(float x, float y) {
// SUCCESS-NEXT:       | ^
// SUCCESS-NEXT: optimization-remarks.c:24:3: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |   ^
// SUCCESS-NEXT: optimization-remarks.c:24:12: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |            ^
// SUCCESS-NEXT: optimization-remarks.c:24:16: remark: instruction promoted to tensor {shape: (Tensor[32], Scalar->Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    24 |   return x * x + y;
// SUCCESS-NEXT:       |                ^
// SUCCESS-NEXT: optimization-remarks.c:30:5: remark: instruction promoted to tensor {shape: (Scalar->Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |     ^
// SUCCESS-NEXT: optimization-remarks.c:30:29: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                             ^
// SUCCESS-NEXT: optimization-remarks.c:30:31: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                               ^
// SUCCESS-NEXT: optimization-remarks.c:30:35: remark: instruction promoted to tensor {shape: (Tensor[32], Scalar->Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                                   ^
// SUCCESS-NEXT: optimization-remarks.c:30:55: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                                                       ^
// SUCCESS-NEXT: optimization-remarks.c:30:57: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    30 |     c[i + ripple_id(BS, 0)] = a[i + ripple_id(BS, 0)] + b[i + ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                                                         ^
// SUCCESS-NEXT: optimization-remarks.c:31:24: remark: branch if-converted to predicated vector form; control flow flattened using mask from vectorized condition {mask-shape: Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    31 |   if (ripple_id(BS, 0) < 4) {
// SUCCESS-NEXT:       |                        ^
// SUCCESS-NEXT: optimization-remarks.c:32:20: remark: scalar call promoted to vector form 'llvm.cos.f32' {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    32 |     float cosVal = __builtin_ripple_cosf(c[ripple_id(BS, 0)]);
// SUCCESS-NEXT:       |                    ^
// SUCCESS:      optimization-remarks.c:33:25: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    33 |     c[ripple_id(BS, 0)] +=
// SUCCESS-NEXT:       |                         ^
// SUCCESS-NEXT: optimization-remarks.c:34:9: remark: scalar call promoted to vector form 'llvm.abs.i32' {shape: (Tensor[32], unchanged) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    34 |         __builtin_abs(ripple_add_sat((int)cosVal, (int)cosVal));
// SUCCESS-NEXT:       |         ^
// SUCCESS-NEXT: optimization-remarks.c:34:23: remark: scalar call promoted to vector form 'llvm.sadd.sat.i32' {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    34 |         __builtin_abs(ripple_add_sat((int)cosVal, (int)cosVal));
// SUCCESS-NEXT:       |                       ^
// SUCCESS:      optimization-remarks.c:36:23: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    36 |   c[ripple_id(BS, 0)] += externf_ew(
// SUCCESS-NEXT:       |                       ^
// SUCCESS-NEXT: optimization-remarks.c:36:26: remark: dispatched call 'externf_ew' to external ripple function 'ripple_ew_pure_externf_ew' {shape: (Tensor[32], Scalar->Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    36 |   c[ripple_id(BS, 0)] += externf_ew(
// SUCCESS-NEXT:       |                          ^
// SUCCESS-NEXT: optimization-remarks.c:37:7: remark: specialized call to 'promotable'; arg0 promoted, arg1 promoted {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    37 |       promotable(externf_exact(c[ripple_id(BS, 0)]), a[ripple_id(BS, 0)]), 1.f);
// SUCCESS-NEXT:       |       ^
// SUCCESS-NEXT: optimization-remarks.c:37:32: remark: instruction promoted to tensor {shape: ({{(Scalar->Tensor\[32\], )?}}Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    37 |       promotable(externf_exact(c[ripple_id(BS, 0)]), a[ripple_id(BS, 0)]), 1.f);
// SUCCESS-NEXT:       |                                ^
// SUCCESS-NEXT: optimization-remarks.c:37:54: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    37 |       promotable(externf_exact(c[ripple_id(BS, 0)]), a[ripple_id(BS, 0)]), 1.f);
// SUCCESS-NEXT:       |                                                      ^
// SUCCESS-NEXT: optimization-remarks.c:39:23: remark: instruction promoted to tensor {shape: (Tensor[32], Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    39 |   c[ripple_id(BS, 0)] += promotable(c[ripple_id(BS, 0)], 42.f);
// SUCCESS-NEXT:       |                       ^
// SUCCESS-NEXT: optimization-remarks.c:39:26: remark: specialized call to 'promotable'; arg0 promoted, arg1 unchanged {shape: (Tensor[32], unchanged) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    39 |   c[ripple_id(BS, 0)] += promotable(c[ripple_id(BS, 0)], 42.f);
// SUCCESS-NEXT:       |                          ^
// SUCCESS-NEXT: optimization-remarks.c:42:24: remark: instruction promoted to tensor {shape: (Tensor[4], Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    42 |   c[ripple_id(BS2, 0)] +=
// SUCCESS-NEXT:       |                        ^
// SUCCESS-NEXT: optimization-remarks.c:43:7: remark: dispatched call 'externf_ew' to external ripple function 'ripple_ew_pure_externf_ew' {shape: (Tensor[4], Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    43 |       externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);
// SUCCESS-NEXT:       |       ^
// SUCCESS-NEXT: optimization-remarks.c:43:18: remark: dispatched call 'externf_exact' to external ripple function 'ripple_externf_exact' {shape: (Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    43 |       externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);
// SUCCESS-NEXT:       |                  ^
// SUCCESS-NEXT: optimization-remarks.c:43:32: remark: instruction promoted to tensor {shape: (Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    43 |       externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);
// SUCCESS-NEXT:       |                                ^
// SUCCESS-NEXT: optimization-remarks.c:43:34: remark: call to ripple API 'llvm.ripple.block.index.i{{32|64}}' {shape: (unchanged, unchanged) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    43 |       externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);
// SUCCESS-NEXT:       |                                  ^
// SUCCESS-NEXT: optimization-remarks.c:43:55: remark: instruction promoted to tensor {shape: (Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    43 |       externf_ew(externf_exact(c[ripple_id(BS2, 0)]), a[ripple_id(BS2, 0)]);
// SUCCESS-NEXT:       |                                                       ^
// SUCCESS-NEXT: optimization-remarks.c:45:24: remark: instruction promoted to tensor {shape: (Tensor[4], Tensor[4]) -> Tensor[4]} [-Rpass=ripple]
// SUCCESS-NEXT:    45 |   c[ripple_id(BS2, 0)] += ripple_reduceadd(0x1, c[ripple_id(BS2, 0)] + 32.f);
// SUCCESS-NEXT:       |                        ^
// SUCCESS-NEXT: optimization-remarks.c:45:27: remark: call to ripple API 'llvm.ripple.reduce.fadd.f32' {shape: (unchanged, Tensor[4]) -> unchanged} [-Rpass=ripple]
// SUCCESS-NEXT:    45 |   c[ripple_id(BS2, 0)] += ripple_reduceadd(0x1, c[ripple_id(BS2, 0)] + 32.f);
// SUCCESS-NEXT:       |                           ^
// SUCCESS:      optimization-remarks.c:48:3: remark: instruction promoted to tensor {shape: (Scalar->Tensor[32][4], Tensor[32][4]) -> Tensor[32][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    48 |   c[ripple_id(BS, 0) + ripple_id(BS3, 1)] =
// SUCCESS-NEXT:       |   ^
// SUCCESS-NEXT: optimization-remarks.c:48:43: remark: instruction promoted to tensor {shape: (Tensor[32][4], Tensor[32][4]) -> Tensor[32][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    48 |   c[ripple_id(BS, 0) + ripple_id(BS3, 1)] =
// SUCCESS-NEXT:       |                                           ^
// SUCCESS-NEXT: optimization-remarks.c:49:7: remark: instruction promoted to tensor {shape: (Tensor[32][4]) -> Tensor[32][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    49 |       a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
// SUCCESS-NEXT:       |       ^
// SUCCESS-NEXT: optimization-remarks.c:49:26: remark: instruction promoted to tensor {shape: (Tensor[1][4]->Tensor[32][4], Tensor[32]->Tensor[32][4]) -> Tensor[32][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    49 |       a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                          ^
// SUCCESS-NEXT: optimization-remarks.c:49:28: remark: call to ripple API 'llvm.ripple.block.index.i{{32|64}}' {shape: (unchanged, unchanged) -> Tensor[1][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    49 |       a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                            ^
// SUCCESS-NEXT: optimization-remarks.c:49:47: remark: instruction promoted to tensor {shape: (Tensor[32][4], Tensor[32]->Tensor[32][4]) -> Tensor[32][4]} [-Rpass=ripple]
// SUCCESS-NEXT:    49 |       a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                                               ^
// SUCCESS-NEXT: optimization-remarks.c:49:49: remark: instruction promoted to tensor {shape: (Tensor[32]) -> Tensor[32]} [-Rpass=ripple]
// SUCCESS-NEXT:    49 |       a[ripple_id(BS, 0) + ripple_id(BS3, 1)] - c[ripple_id(BS, 0)];
// SUCCESS-NEXT:       |                                                 ^
//
// FAILURE:      optimization-remarks.c:37:18: remark: call not vectorized: no suitable vector implementation found for 'externf_exact'; falling back to scalar loop {shape: (Tensor[32]) -> Tensor[32]} [-Rpass-missed=ripple]
// FAILURE-NEXT:    37 |       promotable(externf_exact(c[ripple_id(BS, 0)]), a[ripple_id(BS, 0)]), 1.f);
// FAILURE-NEXT:       |                  ^
