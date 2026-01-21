// Testfile to verify Sema diagnostics for BCD builtins bcdshift, bcdshiftround, bcdtruncate.

// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64le-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc-unknown-unknown -fsyntax-only -verify %s

#include <altivec.h>
#define DECL_COMMON_VARS            \
  vector unsigned char vec = {1,2,3,4}; \
  unsigned char scalar = 1;         \
  int i = 1;                        \
  float f = 1.0f;

vector unsigned char test_bcdshift(void) {
  DECL_COMMON_VARS
  vector unsigned char res_a = __builtin_ppc_bcdshift(scalar, i, i); // expected-error {{argument 0 must be of type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  vector unsigned char res_b = __builtin_ppc_bcdshift(vec, f, i); // expected-error {{argument 1 must be of type integer}}
  vector unsigned char res_c =  __builtin_ppc_bcdshift(vec, i, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_d = __builtin_ppc_bcdshift(vec, i, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_ppc_bcdshift(vec, i, 1);
}

vector unsigned char test_bcdshiftround(void) {
  DECL_COMMON_VARS
  vector unsigned char res_a = __builtin_ppc_bcdshiftround(scalar, i, i); // expected-error {{argument 0 must be of type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  vector unsigned char res_b = __builtin_ppc_bcdshiftround(vec, f, i); // expected-error {{argument 1 must be of type integer}}
  vector unsigned char res_c = __builtin_ppc_bcdshiftround(vec, i, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_d = __builtin_ppc_bcdshiftround(vec, i, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_ppc_bcdshiftround(vec, i, 1);
}

vector unsigned char test_bcdtruncate(void) {
  DECL_COMMON_VARS
  vector unsigned char res_a =  __builtin_ppc_bcdtruncate(scalar, i, i); // expected-error {{argument 0 must be of type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  vector unsigned char res_b =  __builtin_ppc_bcdtruncate(vec, f, i); // expected-error {{argument 1 must be of type integer}}
  vector unsigned char res_c =  __builtin_ppc_bcdtruncate(vec, i, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_d =  __builtin_ppc_bcdtruncate(vec, i, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return  __builtin_ppc_bcdtruncate(vec, i, 1);
}
