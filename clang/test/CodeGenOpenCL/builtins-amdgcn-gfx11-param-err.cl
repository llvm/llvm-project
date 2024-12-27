// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -verify -emit-llvm -o - %s

typedef unsigned int uint;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));

kernel void builtins_amdgcn_bvh_err(global uint2* out, uint addr, uint data, uint4 data1, uint offset) {
  *out = __builtin_amdgcn_ds_bvh_stack_rtn(addr, data, data1, offset); // expected-error {{'__builtin_amdgcn_ds_bvh_stack_rtn' must be a constant integer}}
}
