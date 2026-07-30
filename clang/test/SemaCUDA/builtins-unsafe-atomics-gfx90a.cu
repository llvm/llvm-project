// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device %s \
// RUN:  -fsyntax-only -verify
// expected-no-diagnostics

#define __device__ __attribute__((device))
typedef __attribute__((address_space(3))) float *LP;

__device__ void test_ds_atomic_add_f32(float *addr, float val) {
  float *rtn;
  *rtn = __builtin_amdgcn_ds_faddf((LP)addr, val, 0, 0, 0);
}
