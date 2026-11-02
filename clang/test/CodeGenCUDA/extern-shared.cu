// NOTE: Do not autogenerate. These checks focus on shared types and accesses.
// REQUIRES: nvptx-registered-target, amdgpu-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

// Non-RDC extern __shared__ declarations retain their declared types in shared
// address space. Nonzero-sized complete types use fixed storage; unsized arrays
// use dynamic shared memory. Check both file-scope and block-scope declarations.

#include "Inputs/cuda.h"

struct SharedData {
  int x;
  float y;
};

// CHECK-DAG: %struct.SharedData = type { i32, float }
// CHECK-DAG: @scalar = external{{.*}} addrspace(3) global i32, align 4
// CHECK-DAG: @data = external{{.*}} addrspace(3) global %struct.SharedData, align 4
// CHECK-DAG: @fixed = external{{.*}} addrspace(3) global [4 x i32], align 4
// CHECK-DAG: @dynamic = external{{.*}} addrspace(3) global [0 x i32], align 4
// CHECK-DAG: @local_scalar = external{{.*}} addrspace(3) global i32, align 4
// CHECK-DAG: @local_data = external{{.*}} addrspace(3) global %struct.SharedData, align 4
// CHECK-DAG: @local_fixed = external{{.*}} addrspace(3) global [4 x i32], align 4
// CHECK-DAG: @local_dynamic = external{{.*}} addrspace(3) global [0 x i32], align 4

extern __shared__ int scalar;
extern __shared__ SharedData data;
extern __shared__ int fixed[4];
extern __shared__ int dynamic[];

// CHECK-LABEL: define{{.*}} void @_Z6kernelv(
__global__ void kernel() {
  // CHECK: store i32 1, ptr addrspacecast (ptr addrspace(3) @scalar to ptr)
  scalar = 1;
  // CHECK: store i32 2, ptr addrspacecast (ptr addrspace(3) @data to ptr)
  data.x = 2;
  // CHECK: store i32 3, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @fixed to ptr), i64 4)
  fixed[1] = 3;
  // CHECK: store i32 4, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @dynamic to ptr), i64 4)
  dynamic[1] = 4;

  extern __shared__ int local_scalar;
  extern __shared__ SharedData local_data;
  extern __shared__ int local_fixed[4];
  extern __shared__ int local_dynamic[];
  // CHECK: store i32 5, ptr addrspacecast (ptr addrspace(3) @local_scalar to ptr)
  local_scalar = 5;
  // CHECK: store i32 6, ptr addrspacecast (ptr addrspace(3) @local_data to ptr)
  local_data.x = 6;
  // CHECK: store i32 7, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @local_fixed to ptr), i64 4)
  local_fixed[1] = 7;
  // CHECK: store i32 8, ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @local_dynamic to ptr), i64 4)
  local_dynamic[1] = 8;
}
