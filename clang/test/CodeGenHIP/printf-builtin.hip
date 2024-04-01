// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -disable-llvm-optzns -mprintf-kind=hostcall -fno-builtin-printf -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefixes=CHECK,HOSTCALL %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -disable-llvm-optzns -mprintf-kind=buffered -fno-builtin-printf -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefixes=CHECK,BUFFERED %s

#define __device__ __attribute__((device))

extern "C" __device__ int printf(const char *format, ...);

// CHECK-LABEL: @_Z4foo1v()
__device__ int foo1() {
  // HOSTCALL: call i64 @__ockl_printf_begin
  // BUFFERED: call ptr addrspace(1) @__printf_alloc
  // CHECK-NOT: call i32 (ptr, ...) @printf
  return __builtin_printf("Hello World\n");
}

// CHECK-LABEL: @_Z4foo2v()
__device__ int foo2() {
  // CHECK: call i32 (ptr, ...) @printf
  return printf("Hello World\n");
}
