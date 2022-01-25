// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -S -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))

struct empty {};

__device__ void callee(empty) { }
__host__ __device__ void caller() { callee(empty{}); }

// FIXME: We should not be attaching the dbg.def call to the addrspacecast, but
// to the alloca directly.

// CHECK: %[[ALLOCA:[^ ]+]] = alloca %struct.empty, align 1, addrspace(5)
// CHECK: %[[ASPACE_CAST:[^ ]+]] = addrspacecast %struct.empty addrspace(5)* %[[ALLOCA]] to %struct.empty*
// CHECK: call void @llvm.dbg.def(metadata ![[#PARAM_FRAGMENT:]], metadata %struct.empty* %[[ASPACE_CAST]]), !dbg
// CHECK-DAG: ![[#PARAM_FRAGMENT]] = distinct !DILifetime(object: ![[#PARAM_DILOCALVAR:]], location: !DIExpr(DIOpReferrer(%struct.empty*), DIOpDeref(%struct.empty)))
// CHECK-DAG: ![[#PARAM_DILOCALVAR]] = !DILocalVariable(arg: 1,
