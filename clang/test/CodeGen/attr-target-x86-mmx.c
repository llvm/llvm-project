// RUN: %clang_cc1 -triple i386-linux-gnu -emit-llvm %s -o - | FileCheck %s
// Picking a cpu that doesn't have sse by default so we can enable it later.

#define __MM_MALLOC_H

#include <x86intrin.h>

void __attribute__((target("sse2"))) shift(__m64 a, __m64 b, int c) {
  _mm_slli_pi16(a, c);
  _mm_slli_pi32(a, c);
  _mm_slli_si64(a, c);

  _mm_srli_pi16(a, c);
  _mm_srli_pi32(a, c);
  _mm_srli_si64(a, c);

  _mm_srai_pi16(a, c);
  _mm_srai_pi32(a, c);
}

// CHECK: "target-features"="+cx8,+mmx,+sse,+sse2,+x87"
