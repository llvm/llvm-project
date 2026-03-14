// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -emit-llvm -disable-llvm-optzns -fno-builtin-printf -fcuda-is-device \
// RUN:   -o - %s | FileCheck  %s

#define __device__ __attribute__((device))

extern "C" __device__ int printf(const char *format, ...);

// CHECK-LABEL: @_Z4foo1v()
__device__ int foo1() {
  // CHECK: call i32 @vprintf
  // CHECK-NOT: call i32 (ptr, ...) @printf
  return __builtin_printf("Hello World\n");
}

// CHECK-LABEL: @_Z4foo2v()
__device__ int foo2() {
  // CHECK: call i32 (ptr, ...) @printf
  return printf("Hello World\n");
}
