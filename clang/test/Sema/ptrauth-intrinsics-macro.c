// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify %s

// expected-no-diagnostics

#include <ptrauth.h>

#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2

extern int dv;

void test(int *dp, int (*fp)(int), int value) {
  dp = ptrauth_strip(dp, VALID_DATA_KEY);
  uintptr_t t0 = ptrauth_blend_discriminator(dp, value);
  t0 = ptrauth_type_discriminator(int (*)(int));
  dp = ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0);
  dp = ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, 0);
  dp = ptrauth_auth_and_resign(dp, VALID_DATA_KEY, dp, VALID_DATA_KEY, dp);
  fp = ptrauth_auth_function(fp, VALID_CODE_KEY, 0);
  dp = ptrauth_auth_data(dp, VALID_DATA_KEY, 0);
  int t1 = ptrauth_string_discriminator("string");
  int t2 = ptrauth_sign_generic_data(dp, 0);
}
