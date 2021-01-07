// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#define N 1000

int test_amdgcn_target_tid_threads() {
// CHECK-LABEL: define weak amdgpu_kernel void @{{.*}}test_amdgcn_target_tid_threads

  int arr[N];

// CHECK: [[NUM_THREADS:%.+]] = call i32 @__kmpc_amdgcn_gpu_num_threads()
// CHECK: sub nuw i32 [[NUM_THREADS]], 64
// CHECK: call i32 @llvm.amdgcn.workitem.id.x()
#pragma omp target
  for (int i = 0; i < N; i++) {
    arr[i] = 1;
  }

  return arr[0];
}

int test_amdgcn_target_tid_threads_simd() {
// CHECK-LABEL: define weak amdgpu_kernel void @{{.*}}test_amdgcn_target_tid_threads_simd

  int arr[N];

// CHECK: [[NUM_THREADS:%.+]] = call i32 @__kmpc_amdgcn_gpu_num_threads()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[NUM_THREADS]], i16 0)
#pragma omp target simd
  for (int i = 0; i < N; i++) {
    arr[i] = 1;
  }
  return arr[0];
}

#endif
