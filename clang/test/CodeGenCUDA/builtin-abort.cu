// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s --check-prefix=DEVICE --implicit-check-not=@abort
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s --check-prefix=HOST --implicit-check-not=@llvm.trap
// RUN: %clang_cc1 -x hip -triple amdgpu9.00-amd-amdhsa -fcuda-is-device \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s --check-prefix=DEVICE --implicit-check-not=@abort
// RUN: %clang_cc1 -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s --check-prefix=HOST --implicit-check-not=@llvm.trap
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:   FileCheck %s --check-prefix=HOST --implicit-check-not=@llvm.trap

#include "Inputs/cuda.h"

namespace gh143911 {
// DEVICE-LABEL: define{{.*}} @_ZN8gh1439111fEv(
// DEVICE: call void @llvm.trap()
// DEVICE-NEXT: unreachable
__global__ void f() {
  __builtin_abort();
}
} // namespace gh143911

extern "C" {
// DEVICE-LABEL: define{{.*}} @device_abort(
// DEVICE: call void @llvm.trap()
// DEVICE-NEXT: unreachable
__device__ void device_abort() {
  __builtin_abort();
}

// DEVICE-LABEL: define{{.*}} @shared_abort(
// DEVICE: call void @llvm.trap()
// DEVICE-NEXT: unreachable
// HOST-LABEL: define{{.*}} @shared_abort(
// HOST: call void @abort()
// HOST-NEXT: unreachable
__host__ __device__ void shared_abort() {
  __builtin_abort();
}

// HOST-LABEL: define{{.*}} @host_abort(
// HOST: call void @abort()
// HOST-NEXT: unreachable
__host__ void host_abort() {
  __builtin_abort();
}
}
