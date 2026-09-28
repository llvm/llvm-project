// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fclangir -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fclangir -fcuda-is-device -mprintf-kind=buffered -emit-llvm %s -o %t-cir-buffered.ll
// RUN: FileCheck --check-prefix=BUFFERED --input-file=%t-cir-buffered.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcuda-is-device -mprintf-kind=buffered -emit-llvm %s -o %t-buffered.ll
// RUN: FileCheck --check-prefix=BUFFERED --input-file=%t-buffered.ll %s

// printf fell through to a library call, which does not exist on the device:
// the reference survived to the device link and failed there as an undefined
// symbol. AMDGPU implements it through the OpenCL printf runtime.

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *, ...);

// LLVM-LABEL: @_Z1pi
// LLVM: call i64 @__ockl_printf_begin(i64
// LLVM: call i64 @__ockl_printf_append_string_n(i64
// LLVM: call i64 @__ockl_printf_append_args(i64
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z1pi
// BUFFERED: call ptr addrspace(1) @__printf_alloc(i32
// BUFFERED: end.block:
// BUFFERED: argpush.block:
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ void p(int v) { printf("%d\n", v); }
