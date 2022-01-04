// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

#define __device__ __attribute__((device))

__device__ void kernel1() {
  // CHECK: call void @llvm.dbg.def(metadata ![[#VAR_X_LT:]], metadata %union.anon addrspace(5)* %[[#ALLOCA:]])
  // CHECK: call void @llvm.dbg.def(metadata ![[#VAR_F_LT:]], metadata %union.anon addrspace(5)* %[[#ALLOCA:]])
  // CHECK: call void @llvm.dbg.def(metadata ![[#VAR_ANON_LT:]], metadata %union.anon addrspace(5)* %[[#ALLOCA:]])
  union { int x; float f; };
}

// CHECK: ![[#SUB:]] = distinct !DISubprogram(name: "kernel1",
// CHECK: ![[#VAR_X:]] = !DILocalVariable(name: "x", scope: ![[#SUB]],{{.*}} flags: DIFlagArtificial)
// CHECK: ![[#VAR_F:]] = !DILocalVariable(name: "f", scope: ![[#SUB]],{{.*}} flags: DIFlagArtificial)
// CHECK: ![[#VAR_ANON:]] = !DILocalVariable(scope: ![[#SUB]],{{.*}} flags: DIFlagArtificial)
// CHECK: ![[#VAR_X_LT]] = distinct !DILifetime(object: ![[#VAR_X]], location: !DIExpr(DIOpReferrer(%union.anon addrspace(5)*), DIOpDeref(%union.anon)))
// CHECK: ![[#VAR_F_LT]] = distinct !DILifetime(object: ![[#VAR_F]], location: !DIExpr(DIOpReferrer(%union.anon addrspace(5)*), DIOpDeref(%union.anon)))
// CHECK: ![[#VAR_ANON_LT]] = distinct !DILifetime(object: ![[#VAR_ANON]], location: !DIExpr(DIOpReferrer(%union.anon addrspace(5)*), DIOpDeref(%union.anon)))
