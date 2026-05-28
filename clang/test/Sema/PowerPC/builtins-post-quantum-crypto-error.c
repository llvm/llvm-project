// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu future \
// RUN:   -target-feature -future-vector -fsyntax-only -verify %s

#include <altivec.h>

void test_mulh_builtins(void) {
  vector signed short vss_a, vss_b;
  vector unsigned short vus_a, vus_b;

  // Test __builtin_altivec_vmulhsh - requires future-vector
  vss_a = __builtin_altivec_vmulhsh(vss_a, vss_b); // expected-error {{'__builtin_altivec_vmulhsh' needs target feature future-vector}}

  // Test __builtin_altivec_vmulhuh - requires future-vector
  vus_a = __builtin_altivec_vmulhuh(vus_a, vus_b); // expected-error {{'__builtin_altivec_vmulhuh' needs target feature future-vector}}

  // Test vec_mulh for signed short - no overload available for short types without __FUTURE_VECTOR__
  vss_a = vec_mulh(vss_a, vss_b); // expected-error {{call to 'vec_mulh' is ambiguous}}
  // expected-note@* 4 {{candidate function}}

  // Test vec_mulh for unsigned short - no overload available for short types without __FUTURE_VECTOR__
  vus_a = vec_mulh(vus_a, vus_b); // expected-error {{call to 'vec_mulh' is ambiguous}}
  // expected-note@* 4 {{candidate function}}
}
