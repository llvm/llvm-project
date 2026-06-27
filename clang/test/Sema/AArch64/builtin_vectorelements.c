// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

__attribute__((target("sve")))
long test_builtin_vectorelements_sve(void) {
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sve2p1")))
long test_builtin_vectorelements_sve2p1(void) {
  return __builtin_vectorelements(svuint8_t);
}

long test_builtin_vectorelements_no_sve(void) {
  // expected-error@+1 {{SVE vector type 'svuint8_t' (aka '__SVUint8_t') cannot be used in a target without sve}}
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sme")))
long test_builtin_vectorelements_sme_streaming(void) __arm_streaming {
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sme2p1")))
long test_builtin_vectorelements_sme2p1_streaming(void) __arm_streaming {
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sme")))
long test_builtin_vectorelements_sme(void) {
  // expected-error@+1 {{SVE vector type 'svuint8_t' (aka '__SVUint8_t') cannot be used in a non-streaming function}}
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sve,sme")))
long test_builtin_vectorelements_sve_sme_streaming_compatible(void) __arm_streaming_compatible {
  return __builtin_vectorelements(svuint8_t);
}

__attribute__((target("sme")))
long test_builtin_vectorelements_sme_streaming_compatible(void) __arm_streaming_compatible {
  // expected-error@+1 {{SVE vector type 'svuint8_t' (aka '__SVUint8_t') cannot be used in a non-streaming function}}
  return __builtin_vectorelements(svuint8_t);
}
