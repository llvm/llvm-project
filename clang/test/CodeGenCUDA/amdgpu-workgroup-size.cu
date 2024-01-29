// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -mcode-object-version=4 -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=PRECOV5 %s


// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=COV5 %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -mcode-object-version=none -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=COVNONE %s

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


// COVNONE-LABEL: test_get_workgroup_size
// COVNONE: load i32, ptr addrspace(4) @__oclc_ABI_version
// COVNONE: [[ABI5_X:%.*]] = icmp sge i32 %{{.*}}, 500
// COVNONE: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// COVNONE: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// COVNONE: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// COVNONE: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// COVNONE: select i1 [[ABI5_X]], ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// COVNONE: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// COVNONE: load i32, ptr addrspace(4) @__oclc_ABI_version
// COVNONE: [[ABI5_Y:%.*]] = icmp sge i32 %{{.*}}, 500
// COVNONE: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// COVNONE: [[GEP_5_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 14
// COVNONE: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// COVNONE: [[GEP_4_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 6
// COVNONE: select i1 [[ABI5_Y]], ptr addrspace(4) [[GEP_5_Y]], ptr addrspace(4) [[GEP_4_Y]]
// COVNONE: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// COVNONE: load i32, ptr addrspace(4) @__oclc_ABI_version
// COVNONE: [[ABI5_Z:%.*]] = icmp sge i32 %{{.*}}, 500
// COVNONE: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// COVNONE: [[GEP_5_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 16
// COVNONE: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// COVNONE: [[GEP_4_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 8
// COVNONE: select i1 [[ABI5_Z]], ptr addrspace(4) [[GEP_5_Z]], ptr addrspace(4) [[GEP_4_Z]]
// COVNONE: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

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
