// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -emit-llvm -o /dev/null -verify
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1 -emit-llvm -o /dev/null -verify

#include <immintrin.h>

__attribute__((target("avx512bw")))
__mmask64 k64_verify_1(__mmask64 a) {
  return _knot_mask64(a); // expected-no-diagnostics
}

__mmask64 k64_verify_2(__mmask64 a) {
  return _knot_mask64(a); // expected-no-diagnostic
}

__attribute__((target("avx512bw")))
__m512d zmm_verify_ok(__m512d a) {
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // expected-no-diagnostic
}

__m512d zmm_error(__m512d a) {
  // CHECK-LABEL: @test_mm512_sqrt_pd
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // noevex-error {{'__builtin_ia32_sqrtpd512' needs target feature avx512f}}
}
