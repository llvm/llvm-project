// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:   -mcode-object-version=4 -DUSER -x hip -o %t_4.bc %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:   -mcode-object-version=5 -DUSER -x hip -o %t_5.bc %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:   -mcode-object-version=6 -DUSER -x hip -o %t_6.bc %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:   -mcode-object-version=none -DDEVICELIB -x hip -o %t_0.bc %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -O3 \
// RUN:   %t_4.bc -mlink-builtin-bitcode %t_0.bc -o - |\
// RUN:   FileCheck -check-prefix=LINKED4 %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -O3 \
// RUN:   %t_5.bc -mlink-builtin-bitcode %t_0.bc -o - |\
// RUN:   FileCheck -check-prefix=LINKED5 %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -O3 \
// RUN:   %t_6.bc -mlink-builtin-bitcode %t_0.bc -o - |\
// RUN:   FileCheck -check-prefix=LINKED6 %s

#include "Inputs/cuda.h"

// LINKED4: @__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 400
// LINKED4-LABEL: bar
// LINKED4-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED4-NOT: icmp sge i32 %{{.*}}, 500
// LINKED4: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED4: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// LINKED4: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED4: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// LINKED4: select i1 false, ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// LINKED4: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED4-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED4-NOT: icmp sge i32 %{{.*}}, 500
// LINKED4: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED4: [[GEP_5_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 14
// LINKED4: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED4: [[GEP_4_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 6
// LINKED4: select i1 false, ptr addrspace(4) [[GEP_5_Y]], ptr addrspace(4) [[GEP_4_Y]]
// LINKED4: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED4-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED4-NOT: icmp sge i32 %{{.*}}, 500
// LINKED4: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED4: [[GEP_5_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 16
// LINKED4: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED4: [[GEP_4_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 8
// LINKED4: select i1 false, ptr addrspace(4) [[GEP_5_Z]], ptr addrspace(4) [[GEP_4_Z]]
// LINKED4: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// LINKED4: "amdgpu_code_object_version", i32 400

// LINKED5: __oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500
// LINKED5-LABEL: bar
// LINKED5-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED5-NOT: icmp sge i32 %{{.*}}, 500
// LINKED5: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED5: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// LINKED5: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED5: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// LINKED5: select i1 true, ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// LINKED5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED5-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED5-NOT: icmp sge i32 %{{.*}}, 500
// LINKED5: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED5: [[GEP_5_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 14
// LINKED5: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED5: [[GEP_4_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 6
// LINKED5: select i1 true, ptr addrspace(4) [[GEP_5_Y]], ptr addrspace(4) [[GEP_4_Y]]
// LINKED5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED5-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED5-NOT: icmp sge i32 %{{.*}}, 500
// LINKED5: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED5: [[GEP_5_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 16
// LINKED5: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED5: [[GEP_4_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 8
// LINKED5: select i1 true, ptr addrspace(4) [[GEP_5_Z]], ptr addrspace(4) [[GEP_4_Z]]
// LINKED5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// LINKED5: "amdgpu_code_object_version", i32 500

// LINKED6: __oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 600
// LINKED6-LABEL: bar
// LINKED6-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED6-NOT: icmp sge i32 %{{.*}}, 500
// LINKED6: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED6: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// LINKED6: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED6: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// LINKED6: select i1 true, ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// LINKED6: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED6-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED6-NOT: icmp sge i32 %{{.*}}, 500
// LINKED6: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED6: [[GEP_5_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 14
// LINKED6: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED6: [[GEP_4_Y:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 6
// LINKED6: select i1 true, ptr addrspace(4) [[GEP_5_Y]], ptr addrspace(4) [[GEP_4_Y]]
// LINKED6: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef

// LINKED6-NOT: load i32, ptr addrspacecast (ptr addrspace(4) @__oclc_ABI_version to ptr), align {{.*}}
// LINKED6-NOT: icmp sge i32 %{{.*}}, 500
// LINKED6: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED6: [[GEP_5_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 16
// LINKED6: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED6: [[GEP_4_Z:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 8
// LINKED6: select i1 true, ptr addrspace(4) [[GEP_5_Z]], ptr addrspace(4) [[GEP_4_Z]]
// LINKED6: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// LINKED6: "amdgpu_code_object_version", i32 600

#ifdef DEVICELIB
__device__ void bar(int *x, int *y, int *z)
{
  *x = __builtin_amdgcn_workgroup_size_x();
  *y = __builtin_amdgcn_workgroup_size_y();
  *z = __builtin_amdgcn_workgroup_size_z();
}
#endif

#ifdef USER
__device__ void bar(int *x, int *y, int *z);
__device__ void foo()
{
  int *x, *y, *z;
  bar(x, y, z);
}
#endif
