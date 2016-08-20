// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -target-cpu tahiti -verify -S -o - %s

// FIXME: We only get one error if the functions are the other order in the
// file.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef unsigned long ulong;
typedef unsigned int uint;

ulong test_s_memrealtime()
{
  return __builtin_amdgcn_s_memrealtime(); // expected-error {{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
}

void test_s_sleep(int x)
{
  __builtin_amdgcn_s_sleep(x); // expected-error {{argument to '__builtin_amdgcn_s_sleep' must be a constant integer}}
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
