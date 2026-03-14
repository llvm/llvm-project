// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1100 \
// RUN:   -verify -S -o - %s

// REQUIRES: amdgpu-registered-target

typedef half  __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;


void test_atomic_fadd(__global half2 *addrh2, __local half2 *addrh2l, half2 xh2,
                      __global short2 *addrs2, __local short2 *addrs2l, short2 xs2,
                      __global float *addrf, float xf) {
  __builtin_amdgcn_flat_atomic_fadd_v2f16(addrh2, xh2); // expected-error{{'__builtin_amdgcn_flat_atomic_fadd_v2f16' needs target feature atomic-flat-pk-add-16-insts}}
  __builtin_amdgcn_flat_atomic_fadd_v2bf16(addrs2, xs2); // expected-error{{'__builtin_amdgcn_flat_atomic_fadd_v2bf16' needs target feature atomic-flat-pk-add-16-insts}}
  __builtin_amdgcn_global_atomic_fadd_v2bf16(addrs2, xs2); // expected-error{{'__builtin_amdgcn_global_atomic_fadd_v2bf16' needs target feature atomic-global-pk-add-bf16-inst}}
  __builtin_amdgcn_global_atomic_fadd_v2f16(addrh2, xh2); // expected-error{{'__builtin_amdgcn_global_atomic_fadd_v2f16' needs target feature atomic-buffer-global-pk-add-f16-insts}}
  __builtin_amdgcn_ds_atomic_fadd_v2bf16(addrs2l, xs2); // expected-error{{'__builtin_amdgcn_ds_atomic_fadd_v2bf16' needs target feature atomic-ds-pk-add-16-insts}}
  __builtin_amdgcn_ds_atomic_fadd_v2f16(addrh2l, xh2); // expected-error{{'__builtin_amdgcn_ds_atomic_fadd_v2f16' needs target feature atomic-ds-pk-add-16-insts}}
}
