// Testfile to verify the semantics and the error handling for BCD builtins national2packed, packed2zoned and zoned2packed.
// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64le-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc-unknown-unknown -fsyntax-only -verify %s

#include <altivec.h>
vector unsigned char test_national2packed(void)
{
  vector unsigned char a = {1,2,3,4};
  vector unsigned char res_a = __builtin_ppc_national2packed(a, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_b = __builtin_ppc_national2packed(a, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_ppc_national2packed(a, 0);
}

vector unsigned char test_packed2zoned(void)
{
  vector unsigned char a = {1,2,3,4};
  vector unsigned char res_a = __builtin_ppc_packed2zoned(a,2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_b = __builtin_ppc_packed2zoned(a, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_ppc_packed2zoned(a,1);
}

vector unsigned char test_zoned2packed(void)
{
  vector unsigned char a = {1,2,3,4};
  vector unsigned char res_a = __builtin_ppc_zoned2packed(a,2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vector unsigned char res_b = __builtin_ppc_zoned2packed(a, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return __builtin_ppc_zoned2packed(a,0);
}