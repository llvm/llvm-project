// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -emit-llvm \
// RUN:   -verify -o - %s


// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;
void test_prng_b32(global uint* out, uint a) {
  *out = __builtin_amdgcn_prng_b32(a); // expected-error{{'__builtin_amdgcn_prng_b32' needs target feature prng-inst}}
}
