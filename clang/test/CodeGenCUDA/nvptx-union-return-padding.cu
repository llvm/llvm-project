// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -O0 -S -o - %s | FileCheck %s

#define __device__ __attribute__((device))

union U {
  unsigned long a;
  struct {
    unsigned short x;
    unsigned long y;
  } s;
};

// CHECK-LABEL: .visible .func (.param .align 8 .b8 func_retval0[16]) _Z1fv()
// CHECK: st.param{{.*}}.b64{{.*}}[func_retval0],
// CHECK: st.param{{.*}}.b64{{.*}}[func_retval0+8],
__device__ __attribute__((noinline)) U f() {
  U u = {0x1122334455667788UL};
  return u;
}
