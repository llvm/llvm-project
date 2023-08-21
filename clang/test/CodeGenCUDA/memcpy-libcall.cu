// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -x cuda -triple nvptx64-nvidia-cuda- -fcuda-is-device \
// RUN:     -O3 -S %s -o - | FileCheck -check-prefix=PTX %s
// RUN: %clang_cc1 -x cuda -triple nvptx64-nvidia-cuda- -fcuda-is-device \
// RUN:     -Os -S %s -o - | FileCheck -check-prefix=PTX %s
#include "Inputs/cuda.h"

// PTX-LABEL: .func _Z12copy_genericPvPKv(
void __device__ copy_generic(void *dest, const void *src) {
  __builtin_memcpy(dest, src, 32);
// PTX:        ld.u8
// PTX:        st.u8
}

// PTX-LABEL: .entry _Z11copy_globalPvS_(
void __global__ copy_global(void *dest, void * src) {
  __builtin_memcpy(dest, src, 32);
// PTX:        ld.global.u8
// PTX:        st.global.u8
}

struct S {
  int data[8];
};

// PTX-LABEL: .entry _Z20copy_param_to_globalP1SS_(
void __global__ copy_param_to_global(S *global, S param) {
  __builtin_memcpy(global, &param, sizeof(S));
// PTX:        ld.param.u32
// PTX:        st.global.u32
}

// PTX-LABEL: .entry _Z19copy_param_to_localPU3AS51SS_(
void __global__ copy_param_to_local(__attribute__((address_space(5))) S *local,
                                    S param) {
  __builtin_memcpy(local, &param, sizeof(S));
// PTX:        ld.param.u32
// PTX:        st.local.u32
}

// PTX-LABEL: .func _Z21copy_local_to_genericP1SPU3AS5S_(
void __device__ copy_local_to_generic(S *generic,
                                     __attribute__((address_space(5))) S *src) {
  __builtin_memcpy(generic, src, sizeof(S));
// PTX:        ld.local.u32
// PTX:        st.u32
}

__shared__ S shared;

// PTX-LABEL: .entry _Z20copy_param_to_shared1S(
void __global__ copy_param_to_shared( S param) {
  __builtin_memcpy(&shared, &param, sizeof(S));
// PTX:        ld.param.u32
// PTX:        st.shared.u32
}

void __device__ copy_shared_to_generic(S *generic) {
  __builtin_memcpy(generic, &shared, sizeof(S));
// PTX:        ld.shared.u32
// PTX:        st.u32
}
