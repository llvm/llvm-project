// REQUIRES: amdgpu-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -fcuda-is-device \
// RUN:     -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:     -emit-llvm -disable-llvm-passes -o - -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

__global__ void kernel1(int a) {}
// CHECK: define{{.*}}@_Z7kernel1i{{.*}}#[[ATTR:[0-9]*]]

// CHECK: attributes #[[ATTR]] = {{.*}}norecurse
