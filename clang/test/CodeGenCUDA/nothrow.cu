// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -fcuda-is-device \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -disable-llvm-passes -o - %s | \
// RUN: FileCheck -check-prefix DEVICE %s

// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions \
// RUN:   -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:  FileCheck -check-prefix HOST %s

#include "Inputs/cuda.h"

__host__ __device__ void f();

// HOST: define void @_Z7host_fnv() [[HOST_ATTR:#[0-9]+]]
void host_fn() { f(); }

// DEVICE: define void @_Z3foov() [[DEVICE_ATTR:#[0-9]+]]
__device__ void foo() {
  // DEVICE: call void @_Z1fv
  f();
}

// DEVICE: define void @_Z12foo_noexceptv() [[DEVICE_ATTR:#[0-9]+]]
__device__ void foo_noexcept() noexcept {
  // DEVICE: call void @_Z1fv
  f();
}

// This is nounwind only on the device side.
// CHECK: define void @_Z3foov() [[DEVICE_ATTR:#[0-9]+]]
__host__ __device__ void bar() { f(); }

// DEVICE: define void @_Z3bazv() [[DEVICE_ATTR:#[0-9]+]]
__global__ void baz() { f(); }

// DEVICE: attributes [[DEVICE_ATTR]] = {
// DEVICE-SAME: nounwind
// HOST: attributes [[HOST_ATTR]] = {
// HOST-NOT: nounwind
// HOST-SAME: }
