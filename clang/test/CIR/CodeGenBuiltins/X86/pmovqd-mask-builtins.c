// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

#include <immintrin.h>

__m256i test_pmovqd_mask(__m512i a, __m256i b, __mmask8 mask) {
  // CIR-LABEL: test_pmovqd_mask
  // CIR: %[[TRUNC:.*]] = cir.cast integral {{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<8 x !s32i>
  // CIR: %[[MASK_VEC:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_VEC]], %[[TRUNC]], {{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s32i>
  return __builtin_ia32_pmovqd512_mask(a, b, mask);
}

__m256i test_pmovqd_maskz(__m512i a, __mmask8 mask) {
  // CIR-LABEL: test_pmovqd_maskz
  // CIR: %[[TRUNC:.*]] = cir.cast integral {{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<8 x !s32i>
  // CIR: %[[MASK_VEC:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_VEC]], %[[TRUNC]], {{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s32i>
  __m256i zero = _mm256_setzero_si256();
  return __builtin_ia32_pmovqd512_mask(a, zero, mask);
}

__m256i test_pmovwb_mask(__m512i a, __m256i b, __mmask32 mask) {
  // CIR-LABEL: test_pmovwb_mask
  // CIR: %[[TRUNC:.*]] = cir.cast integral {{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<32 x !s8i>
  // CIR: %[[MASK_VEC:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_VEC]], %[[TRUNC]], {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s8i>
  return __builtin_ia32_pmovwb512_mask(a, b, mask);
}