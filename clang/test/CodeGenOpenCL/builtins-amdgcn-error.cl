// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -target-cpu tahiti -verify -S -o - %s

// FIXME: We only get one error if the functions are the other order in the
// file.

typedef unsigned long ulong;

ulong test_s_memrealtime()
{
  return __builtin_amdgcn_s_memrealtime(); // expected-error {{'__builtin_amdgcn_s_memrealtime' needs target feature s-memrealtime}}
}

void test_s_sleep(int x)
{
  __builtin_amdgcn_s_sleep(x); // expected-error {{argument to '__builtin_amdgcn_s_sleep' must be a constant integer}}
}

