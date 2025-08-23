// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-linux-gnu -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-pc-linux-gnu -aux-triple amdgcn -Wno-unused-value %s
#define __device__ __attribute__((device))

typedef short short2 __attribute__((ext_vector_type(2)));
typedef int int2 __attribute__((ext_vector_type(2)));

struct A { int a; };

__device__ void test_invalid_num_args(int a, int b, int c) {
   __builtin_amdgcn_ds_bpermute(a, b, c); // expected-error {{too many arguments to function call, expected 2, have 3}}
}

__device__ void test_invalid_index(short2 a, int b) {
   __builtin_amdgcn_ds_bpermute(a, b); // expected-error {{converting 'short2' (vector of 2 'short' values) to incompatible type 'int'}}
}

__device__ void test_warn_long_index(long a, int b) {
   __builtin_amdgcn_ds_bpermute(a, b); // expected-warning {{the 64-bit argument will be truncated to 32 bits in this call to __builtin_amdgcn_ds_bpermute}}
}

__device__ void test_vector_src(int a, int2 b) {
   __builtin_amdgcn_ds_bpermute(a, b);
}

__device__ void test_long_src(int a, long b) {
   __builtin_amdgcn_ds_bpermute(a, b);
}

__device__ void test_pointer_src(int a, void* b) {
   __builtin_amdgcn_ds_bpermute(a, b);
}

__device__ void test_struct_src(int a, A b) {
   __builtin_amdgcn_ds_bpermute(a, b);
}
