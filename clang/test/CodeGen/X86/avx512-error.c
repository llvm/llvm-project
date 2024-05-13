// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -target-feature -evex512 -emit-llvm -o /dev/null -verify=noevex
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -emit-llvm -o /dev/null -verify
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1-256 -emit-llvm -o /dev/null -verify=noevex
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1-512 -emit-llvm -o /dev/null -verify

#include <immintrin.h>

// No error emitted whether we have "evex512" feature or not.
__attribute__((target("avx512bw,no-evex512")))
__mmask64 k64_verify_1(__mmask64 a) {
  return _knot_mask64(a); // expected-no-diagnostics
}

__mmask64 k64_verify_2(__mmask64 a) {
  return _knot_mask64(a); // expected-no-diagnostic
}

__attribute__((target("avx512bw,evex512")))
__m512d zmm_verify_ok(__m512d a) {
  // No error emitted if we have "evex512" feature.
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // expected-no-diagnostic
}

__m512d zmm_error(__m512d a) {
  // CHECK-LABEL: @test_mm512_sqrt_pd
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // noevex-error {{'__builtin_ia32_sqrtpd512' needs target feature avx512f,evex512}}
}
#if defined(__AVX10_1__) && !defined(__AVX10_1_512__)
// noevex-warning@*:* {{invalid feature combination: +avx512bw +avx10.1-256; will be promoted to avx10.1-512}}
// noevex-warning@*:* {{invalid feature combination: +avx512bw +avx10.1-256; will be promoted to avx10.1-512}}
// noevex-warning@*:* {{invalid feature combination: +avx512bw +avx10.1-256; will be promoted to avx10.1-512}}
#endif
