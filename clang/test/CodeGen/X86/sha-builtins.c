// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +sha -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

__m128i test_sha1rnds4(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha1rnds4
  return _mm_sha1rnds4_epu32(a, b, 3);
}
__m128i test_sha1nexte(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha1nexte
  return _mm_sha1nexte_epu32(a, b);
}
__m128i test_sha1msg1(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha1msg1
  return _mm_sha1msg1_epu32(a, b);
}
__m128i test_sha1msg2(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha1msg2
  return _mm_sha1msg2_epu32(a, b);
}
__m128i test_sha256rnds2(__m128i a, __m128i b, __m128i c) {
  // CHECK: call <4 x i32> @llvm.x86.sha256rnds2
  return _mm_sha256rnds2_epu32(a, b, c);
}
__m128i test_sha256msg1(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha256msg1
  return _mm_sha256msg1_epu32(a, b);
}
__m128i test_sha256msg2(__m128i a, __m128i b) {
  // CHECK: call <4 x i32> @llvm.x86.sha256msg2
  return _mm_sha256msg2_epu32(a, b);
}
