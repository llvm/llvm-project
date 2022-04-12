// REQUIRES: amdgpu-registered-target

// RUN: %clang -Xclang -no-opaque-pointers -target x86_64-unknown-linux-gnu --offload-arch=gfx906 --cuda-device-only -nogpulib -nogpuinc -x hip -emit-llvm -S -o - %s \
// RUN:   -fgpu-rdc -O3 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false | \
// RUN:   FileCheck %s

#include "Inputs/cuda.h"

// CHECK: %struct.B = type { i8 }
struct B {

  // CHECK: @_ZN1BC1Ei = hidden unnamed_addr alias void (%struct.B*, i32), void (%struct.B*, i32)* @_ZN1BC2Ei
  __device__ B(int x);
};

__device__ B::B(int x) {
}
