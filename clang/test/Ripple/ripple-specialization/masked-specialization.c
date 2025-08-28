// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>
#include <stdint.h>

__attribute__((noinline)) int toBeSpecialized(int32_t n) {
  return n * 32;
}

// This specialization is not using the mask, we can call the unmasked specialization!
// CHECK-LABEL: void @test1
// CHECK-NOT: tail call fastcc <8 x i32> @masked.ripple.specialize
void test1(int32_t *in, int32_t *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);
  size_t idx = ripple_id(BS, 0);
  int x = 34;
  if (idx < 4)
    x = toBeSpecialized(in[idx]);
  out[idx] = x;
}

__attribute__((noinline)) int toBeSpecialized2(int32_t n) {
  return ripple_reduceadd(0x1, (n * 32));
}

// CHECK-LABEL: void @test2
// CHECK: tail call fastcc i32 @masked.ripple.specialization.final.{{[0-9]+}}.toBeSpecialized2(<8 x i32> %{{.*}}, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>)
void test2(int32_t *in, int32_t *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);
  size_t idx = ripple_id(BS, 0);
  int x = 34;
  if (idx < 4)
    x = toBeSpecialized2(in[idx]);
  out[0] = x;
}

__attribute__((noinline)) int toBeSpecialized3(int32_t n, int32_t m) {
  return ripple_reduceadd(0b10, n * m) * 32;
}

// CHECK-LABEL: void @test3
// CHECK: tail call fastcc <2 x i32> @masked.ripple.specialization.final.{{[0-9]+}}.toBeSpecialized3(<2 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>)
void test3(int32_t *in, int32_t *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 2, 4);
  size_t idx = ripple_id(BS, 0);
  size_t idx2 = ripple_id(BS, 1);
  int x = 34;
  if (idx < 2 & idx2 < 2)
    x = toBeSpecialized3(in[idx], in[ripple_get_block_size(BS,0) + idx2]);
  out[idx] = x;
}

__attribute__((noinline)) int toBeSpecialized4(int32_t n, int32_t m) {
  return  n * m * 32;
}

// CHECK-LABEL: void @test4
// CHECK-not: tail call fastcc <2 x i32> @masked.ripple.specialization.final.{{[0-9]+}}.toBeSpecialized4
void test4(int32_t *in, int32_t *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 2, 4);
  size_t idx = ripple_id(BS, 0);
  size_t idx2 = ripple_id(BS, 1);
  int x = 34;
  if (idx < 2 & idx2 < 2)
    x = toBeSpecialized4(in[idx], in[ripple_get_block_size(BS,0) + idx2]);
  out[idx + 2 * idx2] = x;
}

// CHECK-LABEL: define internal fastcc{{.*}}<2 x i32> @masked.ripple.specialization.final.{{[0-9]+}}.toBeSpecialized3
// CHECK-SAME: (<2 x i32> [[Vec0:%.*]], <4 x i32> [[Vec1:%.*]], <8 x i1> [[Mask:%.*]])
// CHECK: [[Arg0Bcast:%.*]] = shufflevector <2 x i32> [[Vec0]], <2 x i32> poison, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
// CHECK-NEXT: [[Arg1Bcast:%.*]] = shufflevector <4 x i32> [[Vec1]], <4 x i32> poison, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
// CHECK-NEXT: [[Mul:%.*]] = mul <8 x i32> [[Arg1Bcast]], [[Arg0Bcast]]
// Masking happens here
// CHECK-NEXT: select <8 x i1> [[Mask]], <8 x i32> [[Mul]], <8 x i32> zeroinitializer
// CHECK: [[MulBy32:%.*]] = shl <2 x i32> %{{.*}}, splat (i32 5)
// CHECK-NEXT: ret <2 x i32> [[MulBy32]]

// CHECK-LABEL: define internal fastcc i32 @masked.ripple.specialization.final.{{[0-9]+}}.toBeSpecialized2
// CHECK-SAME: (<8 x i32> [[Vec:%.*]], <8 x i1> [[Mask:%.*]])
// CHECK: [[Mul:%.*]] = shl <8 x i32> [[Vec]], splat (i32 5)
// Masking happens here
// CHECK-NEXT: select <8 x i1> [[Mask]], <8 x i32> [[Mul]], <8 x i32> zeroinitializer
