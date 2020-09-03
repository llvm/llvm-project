// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: test_get_workgroup_size
// CHECK: call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i32 4
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i32 6
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load
// CHECK: getelementptr i8, i8 addrspace(4)* %{{.*}}, i32 8
// CHECK: load i16, i16 addrspace(4)* %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load
__device__ void test_get_workgroup_size(int d, int *out)
{
  switch (d) {
  case 0: *out = __builtin_amdgcn_workgroup_size_x(); break;
  case 1: *out = __builtin_amdgcn_workgroup_size_y(); break;
  case 2: *out = __builtin_amdgcn_workgroup_size_z(); break;
  default: *out = 0;
  }
}

// CHECK-DAG: [[$WS_RANGE]] = !{i16 1, i16 1025}
