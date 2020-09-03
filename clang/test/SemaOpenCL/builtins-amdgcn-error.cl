// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned long ulong;
typedef unsigned int uint;

// To get all errors for feature checking we need to put them in one function
// since Clang will stop codegen for the next function if it finds error during
// codegen of the previous function.
void test_target_builtin(global int* out, int a)
{
  __builtin_amdgcn_s_memrealtime(); // expected-error {{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
  *out = __builtin_amdgcn_mov_dpp(a, 0, 0, 0, false); // expected-error {{'__builtin_amdgcn_mov_dpp' needs target feature dpp}}
}

void test_s_sleep(int x)
{
  __builtin_amdgcn_s_sleep(x); // expected-error {{argument to '__builtin_amdgcn_s_sleep' must be a constant integer}}
}

void test_s_waitcnt(int x)
{
  __builtin_amdgcn_s_waitcnt(x); // expected-error {{argument to '__builtin_amdgcn_s_waitcnt' must be a constant integer}}
}

void test_s_sendmsg(int in)
{
  __builtin_amdgcn_s_sendmsg(in, 1); // expected-error {{argument to '__builtin_amdgcn_s_sendmsg' must be a constant integer}}
}

void test_s_sendmsg_var(int in1, int in2)
{
  __builtin_amdgcn_s_sendmsg(in1, in2); // expected-error {{argument to '__builtin_amdgcn_s_sendmsg' must be a constant integer}}
}

void test_s_sendmsghalt(int in)
{
  __builtin_amdgcn_s_sendmsghalt(in, 1); // expected-error {{argument to '__builtin_amdgcn_s_sendmsghalt' must be a constant integer}}
}

void test_s_sendmsghalt_var(int in1, int in2)
{
  __builtin_amdgcn_s_sendmsghalt(in1, in2); // expected-error {{argument to '__builtin_amdgcn_s_sendmsghalt' must be a constant integer}}
}

void test_s_incperflevel(int x)
{
  __builtin_amdgcn_s_incperflevel(x); // expected-error {{argument to '__builtin_amdgcn_s_incperflevel' must be a constant integer}}
}

void test_s_decperflevel(int x)
{
  __builtin_amdgcn_s_decperflevel(x); // expected-error {{argument to '__builtin_amdgcn_s_decperflevel' must be a constant integer}}
}

void test_sicmp_i32(global ulong* out, int a, int b, uint c)
{
  *out = __builtin_amdgcn_sicmp(a, b, c); // expected-error {{argument to '__builtin_amdgcn_sicmp' must be a constant integer}}
}

void test_uicmp_i32(global ulong* out, uint a, uint b, uint c)
{
  *out = __builtin_amdgcn_uicmp(a, b, c); // expected-error {{argument to '__builtin_amdgcn_uicmp' must be a constant integer}}
}

void test_sicmp_i64(global ulong* out, long a, long b, uint c)
{
  *out = __builtin_amdgcn_sicmpl(a, b, c); // expected-error {{argument to '__builtin_amdgcn_sicmpl' must be a constant integer}}
}

void test_uicmp_i64(global ulong* out, ulong a, ulong b, uint c)
{
  *out = __builtin_amdgcn_uicmpl(a, b, c); // expected-error {{argument to '__builtin_amdgcn_uicmpl' must be a constant integer}}
}

void test_fcmp_f32(global ulong* out, float a, float b, uint c)
{
  *out = __builtin_amdgcn_fcmpf(a, b, c); // expected-error {{argument to '__builtin_amdgcn_fcmpf' must be a constant integer}}
}

void test_fcmp_f64(global ulong* out, double a, double b, uint c)
{
  *out = __builtin_amdgcn_fcmp(a, b, c); // expected-error {{argument to '__builtin_amdgcn_fcmp' must be a constant integer}}
}

void test_ds_swizzle(global int* out, int a, int b)
{
  *out = __builtin_amdgcn_ds_swizzle(a, b); // expected-error {{argument to '__builtin_amdgcn_ds_swizzle' must be a constant integer}}
}

void test_s_getreg(global int* out, int a)
{
  *out = __builtin_amdgcn_s_getreg(a); // expected-error {{argument to '__builtin_amdgcn_s_getreg' must be a constant integer}}
}

void test_mov_dpp2(global int* out, int a, int b, int c, int d, bool e)
{
  *out = __builtin_amdgcn_mov_dpp(a, b, 0, 0, false); // expected-error {{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(a, 0, c, 0, false); // expected-error {{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(a, 0, 0, d, false); // expected-error {{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(a, 0, 0, 0, e); // expected-error {{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
}

void test_update_dpp2(global int* out, int a, int b, int c, int d, int e, bool f)
{
  *out = __builtin_amdgcn_update_dpp(a, b, 0, 0, 0, false);
  *out = __builtin_amdgcn_update_dpp(a, 0, c, 0, 0, false); // expected-error {{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(a, 0, 0, d, 0, false); // expected-error {{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(a, 0, 0, 0, e, false); // expected-error {{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(a, 0, 0, 0, 0, f); // expected-error {{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
}

void test_ds_faddf(local float *out, float src, int a) {
  *out = __builtin_amdgcn_ds_faddf(out, src, a, 0, false); // expected-error {{argument to '__builtin_amdgcn_ds_faddf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_faddf(out, src, 0, a, false); // expected-error {{argument to '__builtin_amdgcn_ds_faddf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_faddf(out, src, 0, 0, a); // expected-error {{argument to '__builtin_amdgcn_ds_faddf' must be a constant integer}}
}

void test_ds_fminf(local float *out, float src, int a) {
  *out = __builtin_amdgcn_ds_fminf(out, src, a, 0, false); // expected-error {{argument to '__builtin_amdgcn_ds_fminf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_fminf(out, src, 0, a, false); // expected-error {{argument to '__builtin_amdgcn_ds_fminf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_fminf(out, src, 0, 0, a); // expected-error {{argument to '__builtin_amdgcn_ds_fminf' must be a constant integer}}
}

void test_ds_fmaxf(local float *out, float src, int a) {
  *out = __builtin_amdgcn_ds_fmaxf(out, src, a, 0, false); // expected-error {{argument to '__builtin_amdgcn_ds_fmaxf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_fmaxf(out, src, 0, a, false); // expected-error {{argument to '__builtin_amdgcn_ds_fmaxf' must be a constant integer}}
  *out = __builtin_amdgcn_ds_fmaxf(out, src, 0, 0, a); // expected-error {{argument to '__builtin_amdgcn_ds_fmaxf' must be a constant integer}}
}

void test_fence() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST + 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE - 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  __builtin_amdgcn_fence(4); // expected-error {{too few arguments to function call, expected 2}}
  __builtin_amdgcn_fence(4, 4, 4); // expected-error {{too many arguments to function call, expected 2}}
  __builtin_amdgcn_fence(3.14, ""); // expected-warning {{implicit conversion from 'double' to 'unsigned int' changes value from 3.14 to 3}}
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, 5); // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const char *'}}
  const char ptr[] = "workgroup";
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ptr); // expected-error {{expression is not a string literal}}
}

void test_s_setreg(int x, int y) {
  __builtin_amdgcn_s_setreg(x, 0); // expected-error {{argument to '__builtin_amdgcn_s_setreg' must be a constant integer}}
  __builtin_amdgcn_s_setreg(x, y); // expected-error {{argument to '__builtin_amdgcn_s_setreg' must be a constant integer}}
}

void test_atomic_inc32() {
  uint val = 17;
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_SEQ_CST + 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_ACQUIRE - 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_inc32(4);                                            // expected-error {{too few arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_inc32(&val, val, 4, 4, 4, 4);                        // expected-error {{too many arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_inc32(&val, val, 3.14, "");                          // expected-warning {{implicit conversion from 'double' to 'unsigned int' changes value from 3.14 to 3}}
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_ACQUIRE, 5);               // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const char *'}}
  const char ptr[] = "workgroup";
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_ACQUIRE, ptr); // expected-error {{expression is not a string literal}}
  int signedVal = 15;
  signedVal = __builtin_amdgcn_atomic_inc32(&signedVal, signedVal, __ATOMIC_ACQUIRE, ""); // expected-warning {{passing '__private int *' to parameter of type 'volatile __private unsigned int *' converts between pointers to integer types with different sign}}
}

void test_atomic_inc64() {
  __UINT64_TYPE__ val = 17;
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_SEQ_CST + 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_ACQUIRE - 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_inc64(4);                                            // expected-error {{too few arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_inc64(&val, val, 4, 4, 4, 4);                        // expected-error {{too many arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_inc64(&val, val, 3.14, "");                          // expected-warning {{implicit conversion from 'double' to 'unsigned int' changes value from 3.14 to 3}}
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_ACQUIRE, 5);               // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const char *'}}
  const char ptr[] = "workgroup";
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_ACQUIRE, ptr); // expected-error {{expression is not a string literal}}
  __INT64_TYPE__ signedVal = 15;
  signedVal = __builtin_amdgcn_atomic_inc64(&signedVal, signedVal, __ATOMIC_ACQUIRE, ""); // expected-warning {{passing '__private long *' to parameter of type 'volatile __private unsigned long *' converts between pointers to integer types with different sign}}
}

void test_atomic_dec32() {
  uint val = 17;
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST + 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_ACQUIRE - 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_dec32(4);                                            // expected-error {{too few arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_dec32(&val, val, 4, 4, 4, 4);                        // expected-error {{too many arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_dec32(&val, val, 3.14, "");                          // expected-warning {{implicit conversion from 'double' to 'unsigned int' changes value from 3.14 to 3}}
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_ACQUIRE, 5);               // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const char *'}}
  const char ptr[] = "workgroup";
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_ACQUIRE, ptr); // expected-error {{expression is not a string literal}}
  int signedVal = 15;
  signedVal = __builtin_amdgcn_atomic_dec32(&signedVal, signedVal, __ATOMIC_ACQUIRE, ""); // expected-warning {{passing '__private int *' to parameter of type 'volatile __private unsigned int *' converts between pointers to integer types with different sign}}
}

void test_atomic_dec64() {
  __UINT64_TYPE__ val = 17;
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST + 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_ACQUIRE - 1, "workgroup"); // expected-warning {{memory order argument to atomic operation is invalid}}
  val = __builtin_amdgcn_atomic_dec64(4);                                            // expected-error {{too few arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_dec64(&val, val, 4, 4, 4, 4);                        // expected-error {{too many arguments to function call, expected 4}}
  val = __builtin_amdgcn_atomic_dec64(&val, val, 3.14, "");                          // expected-warning {{implicit conversion from 'double' to 'unsigned int' changes value from 3.14 to 3}}
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_ACQUIRE, 5);               // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const char *'}}
  const char ptr[] = "workgroup";
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_ACQUIRE, ptr); // expected-error {{expression is not a string literal}}
  __INT64_TYPE__ signedVal = 15;
  signedVal = __builtin_amdgcn_atomic_dec64(&signedVal, signedVal, __ATOMIC_ACQUIRE, ""); // expected-warning {{passing '__private long *' to parameter of type 'volatile __private unsigned long *' converts between pointers to integer types with different sign}}
}
