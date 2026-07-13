
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -E -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -save-temps=cwd %s -o %t-openmp-amdgcn-amd-amdhsa-gfx90a.i
// RUN: %clang_cc1 -fopenmp  -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -save-temps=cwd -emit-llvm-bc %s -o %t-x86_64-unknown-unknown.bc
// RUN: %clang_cc1 -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -save-temps=cwd -emit-llvm -fopenmp-is-target-device -x cpp-output %t-openmp-amdgcn-amd-amdhsa-gfx90a.i -fopenmp-host-ir-file-path %t-x86_64-unknown-unknown.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#define N 1000

int test_amdgcn_save_temps() {
  int arr[N];
#pragma omp target
  for (int i = 0; i < N; i++) {
    arr[i] = 1;
  }
  return arr[0];
}
#endif

// CHECK: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}_test_amdgcn_save_temps
