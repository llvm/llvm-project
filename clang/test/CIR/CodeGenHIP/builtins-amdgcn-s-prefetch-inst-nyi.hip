// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1250 -fcuda-is-device -emit-cir %s -verify -o %t.cir

#define __device__ __attribute__((device))

// expected-error@+2 {{ClangIR code gen Not Yet Implemented: unimplemented AMDGPU builtin call: __builtin_amdgcn_s_prefetch_inst}}
__device__ void test_s_prefetch_inst(const void *p, unsigned int len) {
  __builtin_amdgcn_s_prefetch_inst(p, len);
}
