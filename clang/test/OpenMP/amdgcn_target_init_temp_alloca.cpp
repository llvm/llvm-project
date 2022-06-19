// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

#define N 100

int test_amdgcn_target_temp_alloca() {
  // CHECK-LABEL: test_amdgcn_target_temp_alloca

  int arr[N];

  // CHECK:      [[VAR_ADDR:%.+]] = alloca [100 x i32]*, align 8, addrspace(5)
  // CHECK-NEXT: [[VAR2_ADDR:%.+]] = alloca i32, align 4, addrspace(5)
  // CHECK-NEXT: [[VAR_ADDR_CAST:%.+]] = addrspacecast [100 x i32]* addrspace(5)* [[VAR_ADDR]] to [100 x i32]**
  // CHECK-NEXT: [[VAR2_ADDR_CAST:%.+]] = addrspacecast i32 addrspace(5)* [[VAR2_ADDR]] to i32*
  // CHECK:  store [100 x i32]* [[VAR:%.+]], [100 x i32]** [[VAR_ADDR_CAST]], align 8

#pragma omp target
  for (int i = 0; i < N; i++) {
    arr[i] = 1;
  }

  return arr[0];
}
