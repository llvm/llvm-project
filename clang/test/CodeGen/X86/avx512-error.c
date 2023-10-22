// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -target-feature -evex512 -emit-llvm -o /dev/null -verify=noevex -DFEATURE_TEST=1
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -target-feature -evex512 -emit-llvm -o /dev/null -verify=noevex -DFEATURE_TEST=2
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512bw -emit-llvm -o /dev/null -verify -DFEATURE_TEST=3
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1-256 -emit-llvm -o /dev/null -verify=noevex -DFEATURE_TEST=1
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1-256 -emit-llvm -o /dev/null -verify=noevex -DFEATURE_TEST=2
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1-512 -emit-llvm -o /dev/null -verify -DFEATURE_TEST=3

#include <immintrin.h>

#if FEATURE_TEST & 3
// expected-no-diagnostics
#endif

#if FEATURE_TEST & 1
__attribute__((target("avx512bw,evex512")))
__m512d zmm_verify_ok(__m512d a) {
  // No error emitted if we have "evex512" feature.
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION);
}

__m512d zmm_error(__m512d a) {
  // CHECK-LABEL: @test_mm512_sqrt_pd
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // noevex-error {{'__builtin_ia32_sqrtpd512' needs target feature avx512f,evex512}}
}
#endif

#if FEATURE_TEST & 2
__attribute__((target("avx512bw,evex512")))
__mmask64 k64_verify_ok(__mmask64 a) {
  // No error emitted if we have "evex512" feature.
  return _knot_mask64(a);
}

__mmask64 test_knot_mask64(__mmask64 a) {
  return _knot_mask64(a); // noevex-error {{always_inline function '_knot_mask64' requires target feature 'evex512', but would be inlined into function 'test_knot_mask64' that is compiled without support for 'evex512'}}
}
#endif
