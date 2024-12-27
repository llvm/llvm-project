// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s -fexperimental-new-constant-interpreter

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

void test_string_discriminator(const char *str) {
  __builtin_ptrauth_string_discriminator(); // expected-error {{too few arguments}}
  __builtin_ptrauth_string_discriminator(str, str); // expected-error {{too many arguments}}
  (void) __builtin_ptrauth_string_discriminator("test string"); // no warning

  __builtin_ptrauth_string_discriminator(str); // expected-error {{argument must be a string literal}}
  __builtin_ptrauth_string_discriminator(L"wide test"); // expected-error {{argument must be a string literal}} expected-warning {{incompatible pointer types passing 'int[10]' to parameter of type 'const char *'}}

  void *mismatch = __builtin_ptrauth_string_discriminator("test string"); // expected-error {{incompatible integer to pointer conversion initializing 'void *' with an expression of type 'unsigned long'}}
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


typedef int (*fp_t)(int);

static int dv_weakref __attribute__((weakref("dv")));
extern int dv_weak __attribute__((weak));

int *t_cst_sig1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY); // expected-error {{too few arguments}}
int *t_cst_sig2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, &dv, &dv); // expected-error {{too many arguments}}

int *t_cst_sig3 = __builtin_ptrauth_sign_constant(mismatched_type, VALID_DATA_KEY, 0); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
int *t_cst_sig4 = __builtin_ptrauth_sign_constant(&dv, mismatched_type, 0); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
int *t_cst_sig5 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

float *t_cst_result = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}

int *t_cst_valid1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0);
int *t_cst_valid2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&dv, 0));
int *t_cst_valid3 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&dv + 8, 0));
int *t_cst_valid4 = __builtin_ptrauth_sign_constant(&dv_weak, VALID_DATA_KEY, 0);
int *t_cst_valid5 = __builtin_ptrauth_sign_constant(&dv_weakref, VALID_DATA_KEY, 0);

int *t_cst_ptr = __builtin_ptrauth_sign_constant(NULL, VALID_DATA_KEY, &dv); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}
int *t_cst_key = __builtin_ptrauth_sign_constant(&dv, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}
int *t_cst_disc1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, &fv); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
int *t_cst_disc2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&fv, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

fp_t t_cst_f_valid1 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, 0);
fp_t t_cst_f_valid2 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&dv, 0));
fp_t t_cst_f_valid3 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&dv + 8, 0));

fp_t t_cst_f_key = __builtin_ptrauth_sign_constant(&fv, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}
fp_t t_cst_f_disc1 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, &fv); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
fp_t t_cst_f_disc2 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&fv, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

int *t_cst_offset = __builtin_ptrauth_sign_constant((int *)((char*)&dv + 16), VALID_DATA_KEY, 0);
fp_t t_cst_f_offset = __builtin_ptrauth_sign_constant((int (*)(int))((char*)&fv + 16), VALID_CODE_KEY, 0); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}

void test_sign_constant(int *dp, fp_t fp) {
  int *sig1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY); // expected-error {{too few arguments}}
  int *sig2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, &dv, &dv); // expected-error {{too many arguments}}

  int *sig3 = __builtin_ptrauth_sign_constant(mismatched_type, VALID_DATA_KEY, 0); // expected-error {{signed value must have pointer type; type here is 'struct A'}}
  int *sig4 = __builtin_ptrauth_sign_constant(&dv, mismatched_type, 0); // expected-error {{passing 'struct A' to parameter of incompatible type 'int'}}
  int *sig5 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, mismatched_type); // expected-error {{extra discriminator must have pointer or integer type; type here is 'struct A'}}

  float *result = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0); // expected-warning {{incompatible pointer types initializing 'float *' with an expression of type 'int *'}}

  int *valid1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0);
  int *valid2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&dv, 0));
  int *valid3 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&dv + 8, 0));
  int *valid4 = __builtin_ptrauth_sign_constant(&dv_weak, VALID_DATA_KEY, 0);
  int *valid5 = __builtin_ptrauth_sign_constant(&dv_weakref, VALID_DATA_KEY, 0);

  int *ptr = __builtin_ptrauth_sign_constant(NULL, VALID_DATA_KEY, &dv); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}
  int *key = __builtin_ptrauth_sign_constant(&dv, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}
  int *disc1 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, &fv); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
  int *disc2 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&fv, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

  int *ptr2 = __builtin_ptrauth_sign_constant(dp, VALID_DATA_KEY, 0); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}
  int *disc3 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, dp); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
  int *disc4 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(dp, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
  int *disc5 = __builtin_ptrauth_sign_constant(&dv, VALID_DATA_KEY, __builtin_ptrauth_blend_discriminator(&dv, *dp)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

  fp_t f_valid1 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, 0);
  fp_t f_valid2 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&dv, 0));
  fp_t f_valid3 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&dv + 8, 0));

  fp_t f_key = __builtin_ptrauth_sign_constant(&fv, INVALID_KEY, 0); // expected-error {{does not identify a valid pointer authentication key for the current target}}
  fp_t f_disc1 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, &fv); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
  fp_t f_disc2 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&fv, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

  fp_t f_ptr = __builtin_ptrauth_sign_constant(fp, VALID_CODE_KEY, 0); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}
  fp_t f_disc3 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, dp); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

  fp_t f_disc4 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(dp, 0)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}
  fp_t f_disc5 = __builtin_ptrauth_sign_constant(&fv, VALID_CODE_KEY, __builtin_ptrauth_blend_discriminator(&dv, *dp)); // expected-error {{discriminator argument to ptrauth_sign_constant must be a constant integer, the address of the global variable where the result will be stored, or a blend of the two}}

  int *offset = __builtin_ptrauth_sign_constant((int *)((char*)&dv + 16), VALID_DATA_KEY, 0);
  fp_t f_offset = __builtin_ptrauth_sign_constant((fp_t)((char*)&fv + 16), VALID_CODE_KEY, 0); // expected-error {{argument to ptrauth_sign_constant must refer to a global variable or function}}
}
