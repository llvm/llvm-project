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

