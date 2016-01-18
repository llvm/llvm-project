// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +f16c -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +f16c -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128 test_mm_cvtph_ps(__m128i a) {
  // CHECK: @llvm.x86.vcvtph2ps.128
  // CHECK-ASM: vcvtph2ps %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtph_ps(a);
}

__m256 test_mm256_cvtph_ps(__m128i a) {
  // CHECK: @llvm.x86.vcvtph2ps.256
  // CHECK-ASM: vcvtph2ps %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtph_ps(a);
}

__m128i test_mm_cvtps_ph(__m128 a) {
  // CHECK: @llvm.x86.vcvtps2ph.128
  // CHECK-ASM: vcvtps2ph $0, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cvtps_ph(a, 0);
}

__m128i test_mm256_cvtps_ph(__m256 a) {
  // CHECK: @llvm.x86.vcvtps2ph.256
  // CHECK-ASM: vcvtps2ph $0, %ymm{{.*}}, %xmm{{.*}}
  return _mm256_cvtps_ph(a, 0);
}
