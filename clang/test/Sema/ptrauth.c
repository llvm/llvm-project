// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s

#if __has_feature(ptrauth_intrinsics)
#warning Pointer authentication enabled!
// expected-warning@-1 {{Pointer authentication enabled!}}
#endif

#if __aarch64__
#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2
#define INVALID_KEY 200
#else
#error Provide these constants if you port this test
#endif

#define NULL ((void*) 0)
struct A { int x; } mismatched_type;

extern int dv;
extern int fv(int);

void test_strip(int *dp, int (*fp)(int)) {
  __builtin_ptrauth_strip(dp); // expected-error {{too few arguments}}
  __builtin_ptrauth_strip(dp, VALID_DATA_KEY, dp); // expected-error {{too many arguments}}
  (void) __builtin_ptrauth_strip(NULL, VALID_DATA_KEY); // no warning

  __builtin_ptrauth_strip(mismatched_type, VALID_DATA_KEY); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
  __builtin_ptrauth_strip(dp, mismatched_type); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}

  int *dr = __builtin_ptrauth_strip(dp, VALID_DATA_KEY);
  dr = __builtin_ptrauth_strip(dp, INVALID_KEY); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  int (*fr)(int) = __builtin_ptrauth_strip(fp, VALID_CODE_KEY);
  fr = __builtin_ptrauth_strip(fp, INVALID_KEY); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  float *mismatch = __builtin_ptrauth_strip(dp, VALID_DATA_KEY); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}
}

void test_blend_discriminator(int *dp, int (*fp)(int), int value) {
  __builtin_ptrauth_blend_discriminator(dp); // expected-error {{too few arguments}}
  __builtin_ptrauth_blend_discriminator(dp, dp, dp); // expected-error {{too many arguments}}
  (void) __builtin_ptrauth_blend_discriminator(dp, value); // no warning

  __builtin_ptrauth_blend_discriminator(mismatched_type, value); // expected-error {{blended pointer must have pointer type; type here is 'struct A'}}
  __builtin_ptrauth_blend_discriminator(dp, mismatched_type); // expected-error {{blended integer must have integer type; type here is 'struct A'}}

  float *mismatch = __builtin_ptrauth_blend_discriminator(dp, value); // expected-error {{incompatible integer to pointer conversion initializing 'float *' with an expression of type}}
}

void test_sign_unauthenticated(int *dp, int (*fp)(int)) {
  __builtin_ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY); // expected-error {{too few arguments}}
  __builtin_ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, dp, dp); // expected-error {{too many arguments}}

  __builtin_ptrauth_sign_unauthenticated(mismatched_type, VALID_DATA_KEY, 0); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
  __builtin_ptrauth_sign_unauthenticated(dp, mismatched_type, 0); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
  __builtin_ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

  (void) __builtin_ptrauth_sign_unauthenticated(NULL, VALID_DATA_KEY, 0); // expected-warning {{signing a null pointer will yield a non-null pointer}}

  int *dr = __builtin_ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, 0);
  dr = __builtin_ptrauth_sign_unauthenticated(dp, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  int (*fr)(int) = __builtin_ptrauth_sign_unauthenticated(fp, VALID_CODE_KEY, 0);
  fr = __builtin_ptrauth_sign_unauthenticated(fp, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  float *mismatch = __builtin_ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, 0); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}
}

void test_auth(int *dp, int (*fp)(int)) {
  __builtin_ptrauth_auth(dp, VALID_DATA_KEY); // expected-error {{too few arguments}}
  __builtin_ptrauth_auth(dp, VALID_DATA_KEY, dp, dp); // expected-error {{too many arguments}}

  __builtin_ptrauth_auth(mismatched_type, VALID_DATA_KEY, 0); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
  __builtin_ptrauth_auth(dp, mismatched_type, 0); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
  __builtin_ptrauth_auth(dp, VALID_DATA_KEY, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

  (void) __builtin_ptrauth_auth(NULL, VALID_DATA_KEY, 0); // expected-warning {{authenticating a null pointer will almost certainly trap}}

  int *dr = __builtin_ptrauth_auth(dp, VALID_DATA_KEY, 0);
  dr = __builtin_ptrauth_auth(dp, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  int (*fr)(int) = __builtin_ptrauth_auth(fp, VALID_CODE_KEY, 0);
  fr = __builtin_ptrauth_auth(fp, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  float *mismatch = __builtin_ptrauth_auth(dp, VALID_DATA_KEY, 0); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}
}

void test_auth_and_resign(int *dp, int (*fp)(int)) {
  __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, VALID_DATA_KEY); // expected-error {{too few arguments}}
  __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, dp, VALID_DATA_KEY, dp, 0); // expected-error {{too many arguments}}

  __builtin_ptrauth_auth_and_resign(mismatched_type, VALID_DATA_KEY, 0, VALID_DATA_KEY, dp); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
  __builtin_ptrauth_auth_and_resign(dp, mismatched_type, 0, VALID_DATA_KEY, dp); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
  __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, mismatched_type, VALID_DATA_KEY, dp); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}
  __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, mismatched_type, dp); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
  __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, VALID_DATA_KEY, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

  (void) __builtin_ptrauth_auth_and_resign(NULL, VALID_DATA_KEY, 0, VALID_DATA_KEY, dp); // expected-warning {{authenticating a null pointer will almost certainly trap}}

  int *dr = __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, VALID_DATA_KEY, dp);
  dr = __builtin_ptrauth_auth_and_resign(dp, INVALID_KEY, 0, VALID_DATA_KEY, dp); // expected-error {{does not identify a valid pointer authentication key for the current target}}
  dr = __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, INVALID_KEY, dp); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  int (*fr)(int) = __builtin_ptrauth_auth_and_resign(fp, VALID_CODE_KEY, 0, VALID_CODE_KEY, dp);
  fr = __builtin_ptrauth_auth_and_resign(fp, INVALID_KEY, 0, VALID_CODE_KEY, dp); // expected-error {{does not identify a valid pointer authentication key for the current target}}
  fr = __builtin_ptrauth_auth_and_resign(fp, VALID_CODE_KEY, 0, INVALID_KEY, dp); // expected-error {{does not identify a valid pointer authentication key for the current target}}

  float *mismatch = __builtin_ptrauth_auth_and_resign(dp, VALID_DATA_KEY, 0, VALID_DATA_KEY, dp); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}
}

void test_sign_generic_data(int *dp) {
  __builtin_ptrauth_sign_generic_data(dp); // expected-error {{too few arguments}}
  __builtin_ptrauth_sign_generic_data(dp, 0, 0); // expected-error {{too many arguments}}

  __builtin_ptrauth_sign_generic_data(mismatched_type, 0); // expected-error {{signed value must have pointer or integer type; type here is 'struct A'}}
  __builtin_ptrauth_sign_generic_data(dp, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

  (void) __builtin_ptrauth_sign_generic_data(NULL, 0); // no warning

  unsigned long dr = __builtin_ptrauth_sign_generic_data(dp, 0);
  dr = __builtin_ptrauth_sign_generic_data(dp, &dv);
  dr = __builtin_ptrauth_sign_generic_data(12314, 0);
  dr = __builtin_ptrauth_sign_generic_data(12314, &dv);

  int *mismatch = __builtin_ptrauth_sign_generic_data(dp, 0); // expected-error {{incompatible integer to pointer conversion initializing 'int *' with an expression of type}}
}
