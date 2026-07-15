// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify=expected,wave64 -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature +wavefrontsize64 -verify=expected,wave64 -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature +wavefrontsize32 -verify=expected,wave32 -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -verify=expected,wave32 -fsyntax-only %s

// Check that -Wno-deprecated-builtins silences the warnings.
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -Wno-deprecated-builtins -verify=silenced -fsyntax-only %s

// REQUIRES: amdgpu-registered-target

// silenced-no-diagnostics

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned int uint;
typedef unsigned long ulong;

void test_sicmp(global ulong* out, int a, int b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_sicmp' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_sicmp' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_sicmp(a, b, 32);
}

void test_sicmpl(global ulong* out, long a, long b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_sicmpl' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_sicmpl' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_sicmpl(a, b, 32);
}

void test_uicmp(global ulong* out, uint a, uint b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_uicmp' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_uicmp' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_uicmp(a, b, 32);
}

void test_uicmpl(global ulong* out, ulong a, ulong b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_uicmpl' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_uicmpl' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_uicmpl(a, b, 32);
}

void test_fcmp(global ulong* out, double a, double b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_fcmp' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_fcmp' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_fcmp(a, b, 1);
}

void test_fcmpf(global ulong* out, float a, float b) {
  // wave64-warning@+2 {{builtin '__builtin_amdgcn_fcmpf' is deprecated; use __builtin_amdgcn_ballot_w64 instead}}
  // wave32-warning@+1 {{builtin '__builtin_amdgcn_fcmpf' is deprecated; use __builtin_amdgcn_ballot_w32 instead}}
  *out = __builtin_amdgcn_fcmpf(a, b, 1);
}
