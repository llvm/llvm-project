// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s
// FIXME: The shufflevector instructions in test_cmpgt_sd are relying on O3 here.


#include <immintrin.h>

//
// Test LLVM IR codegen of cmpXY instructions
//

__m128d test_cmp_sd(__m128d a, __m128d b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 13)
  return _mm_cmp_sd(a, b, _CMP_GE_OS);
}

__m128 test_cmp_ss(__m128 a, __m128 b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 13)
  return _mm_cmp_ss(a, b, _CMP_GE_OS);
}

__m128 test_cmpgt_ss(__m128 a, __m128 b) {
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 1)
  // CHECK: shufflevector <{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  return _mm_cmpgt_ss(a, b);
}

__m128 test_cmpge_ss(__m128 a, __m128 b) {
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 2)
  // CHECK: shufflevector <{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  return _mm_cmpge_ss(a, b);
}

__m128 test_cmpngt_ss(__m128 a, __m128 b) {
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 5)
  // CHECK: shufflevector <{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  return _mm_cmpngt_ss(a, b);
}

__m128 test_cmpnge_ss(__m128 a, __m128 b) {
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 6)
  // CHECK: shufflevector <{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  return _mm_cmpnge_ss(a, b);
}

__m128d test_cmpgt_sd(__m128d a, __m128d b) {
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 1)
  // CHECK: shufflevector <{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_cmpgt_sd(a, b);
}

__m128d test_cmpge_sd(__m128d a, __m128d b) {
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 2)
  // CHECK: shufflevector <{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_cmpge_sd(a, b);
}

__m128d test_cmpngt_sd(__m128d a, __m128d b) {
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 5)
  // CHECK: shufflevector <{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_cmpngt_sd(a, b);
}

__m128d test_cmpnge_sd(__m128d a, __m128d b) {
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 6)
  // CHECK: shufflevector <{{.*}}, <2 x i32> <i32 0, i32 3>
  return _mm_cmpnge_sd(a, b);
}
