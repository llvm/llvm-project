// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

// This test is to verify that exec_mode of 0 is passed down from compiler to runtime.
// Based on existing test offload/test/offloading/ompx_bare.c.

#include <ompx.h>

int foo(int *data) {
  const int num_blocks = 64;
  const int block_size = 64;
  const int N = num_blocks * block_size;

#pragma omp target teams ompx_bare num_teams(num_blocks) thread_limit(block_size) map(from: data[0:N])
  {
    int bid = ompx_block_id_x();
    int bdim = ompx_block_dim_x();
    int tid = ompx_thread_id_x();
    int idx = bid * bdim + tid;
    data[idx] = idx;
  }

  return 0;
}
// CHECK-DAG: @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooPi_l15_exec_mode = weak addrspace(1) constant i8 0

// CHECK-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooPi_l15
// CHECK-SAME: (ptr noalias noundef [[DYN_PTR:%.*]], ptr noundef [[DATA:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK-NEXT:    [[DATA_ADDR:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK-NEXT:    [[DOTZERO_ADDR:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[DYN_PTR_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DYN_PTR_ADDR]] to ptr
// CHECK-NEXT:    [[DATA_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DATA_ADDR]] to ptr
// CHECK-NEXT:    [[DOTZERO_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTZERO_ADDR]] to ptr
// CHECK-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR_ASCAST]], align 8
// CHECK-NEXT:    store ptr [[DATA]], ptr [[DATA_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[DATA_ADDR_ASCAST]], align 8
// CHECK-NEXT:    store i32 0, ptr [[DOTZERO_ADDR_ASCAST]], align 4
// CHECK-NEXT:    call void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooPi_l15_omp_outlined(ptr null, ptr [[DOTZERO_ADDR_ASCAST]], ptr [[TMP0]]) #[[ATTR4:[0-9]+]]
// CHECK-NEXT:    ret void
//
