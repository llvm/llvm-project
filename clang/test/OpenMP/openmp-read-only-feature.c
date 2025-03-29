// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: clang-target-64-bits

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -fopenmp -nogpulib -fopenmp-is-target-device -verify %s
// expected-no-diagnostics

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -fopenmp -nogpulib -target-feature -image-insts \
// RUN:   -fopenmp-is-target-device -S -o - %s 2>&1 | FileCheck %s
// CHECK: warning: feature flag '-image-insts' is ignored since the feature is read only

#pragma omp begin declare variant match(device = {arch(amdgcn)})
void foo();
#pragma omp end declare variant
