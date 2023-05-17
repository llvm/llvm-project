// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=PRECOV5 %s


// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -mcode-object-version=5 -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=COV5 %s

#include "Inputs/cuda.h"

// PRECOV5-LABEL: test_get_workgroup_size
// PRECOV5: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// PRECOV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// PRECOV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// PRECOV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 6
// PRECOV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// PRECOV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 8
// PRECOV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// COV5-LABEL: test_get_workgroup_size
// COV5: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// COV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// COV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// COV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 14
// COV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// COV5: getelementptr i8, ptr addrspace(4) %{{.*}}, i32 16
// COV5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
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
