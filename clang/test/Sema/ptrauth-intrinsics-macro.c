// RUN: %clang_cc1 -triple arm64-apple-ios -Wall -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple arm64-apple-ios -Wall -fsyntax-only -verify %s

// expected-no-diagnostics

#include <ptrauth.h>

#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2

extern int dv;

void test(int *dp, int value) {
  dp = ptrauth_strip(dp, VALID_DATA_KEY);
  ptrauth_extra_data_t t0 = ptrauth_blend_discriminator(dp, value);
  (void)t0;
  dp = ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, 0);
  dp = ptrauth_auth_and_resign(dp, VALID_DATA_KEY, dp, VALID_DATA_KEY, dp);
  dp = ptrauth_auth_data(dp, VALID_DATA_KEY, 0);
  int pu0 = 0, pu1 = 0, pu2 = 0, pu3 = 0, pu4 = 0, pu5 = 0, pu6 = 0, pu7 = 0;
  ptrauth_blend_discriminator(&pu0, value);
  ptrauth_auth_and_resign(&pu1, VALID_DATA_KEY, dp, VALID_DATA_KEY, dp);
  ptrauth_auth_and_resign(dp, VALID_DATA_KEY, &pu2, VALID_DATA_KEY, dp);
  ptrauth_auth_and_resign(dp, VALID_DATA_KEY, dp, VALID_DATA_KEY, &pu3);
  ptrauth_sign_generic_data(pu4, dp);
  ptrauth_sign_generic_data(dp, pu5);
  ptrauth_auth_data(&pu6, VALID_DATA_KEY, value);
  ptrauth_auth_data(dp, VALID_DATA_KEY, pu7);



  int t2 = ptrauth_sign_generic_data(dp, 0);
  (void)t2;
}

void test_string_discriminator(int *dp) {
  ptrauth_extra_data_t t0 = ptrauth_string_discriminator("string");
  (void)t0;
}

void test_type_discriminator(int *dp) {
  ptrauth_extra_data_t t0 = ptrauth_type_discriminator(int (*)(int));
  (void)t0;
}

void test_sign_constant(int *dp) {
  dp = ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0);
}
