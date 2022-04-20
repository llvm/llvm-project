// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// CHECK-LABEL: @_Z7kernel1i
// CHECK-SAME: !dbg ![[#FUNC_SUBPROGRAM:]]
// CHECK: call void @llvm.dbg.def(metadata ![[#ARG_LIFETIME:]], metadata ptr addrspace(5) %Arg.addr)
__device__ void kernel1(int Arg) {

  __shared__ int FuncVarShared;

  // CHECK: call void @llvm.dbg.def(metadata ![[#FUNC_VAR_LIFETIME:]], metadata ptr addrspace(5) %FuncVar)
  int FuncVar;
}

// CHECK-LABEL: !llvm.dbg.retainedNodes =
// CHECK-SAME: !{![[#FUNC_VAR_SHARED_LIFETIME:]]}

// CHECK-DAG: ![[#FUNC_SUBPROGRAM]] = distinct !DISubprogram(name: "kernel1"{{.*}}retainedNodes: ![[#FUNC_RETAINED_NODES:]]{{.*}})
// CHECK-DAG: ![[#FUNC_RETAINED_NODES]] = !{![[#ARG:]], ![[#FUNC_VAR:]]}

// CHECK-DAG: ![[#FUNC_VAR_SHARED_LIFETIME]] = distinct !DILifetime(object: ![[#FUNC_VAR_SHARED:]], location: !DIExpr(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)), argObjects: {![[#FUNC_VAR_SHARED_FRAGMENT:]]})
// CHECK-DAG: ![[#FUNC_VAR_SHARED]] = distinct !DIGlobalVariable(name: "FuncVarShared",
// CHECK-DAG: ![[#FUNC_VAR_SHARED_FRAGMENT]] = distinct !DIFragment()

// CHECK-DAG: ![[#ARG_LIFETIME]] = distinct !DILifetime(object: ![[#ARG]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
// CHECK-DAG: ![[#ARG]] = !DILocalVariable(name: "Arg", arg: 1,

// CHECK-DAG: ![[#FUNC_VAR_LIFETIME]] = distinct !DILifetime(object: ![[#FUNC_VAR]], location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
// CHECK-DAG: ![[#FUNC_VAR]] = !DILocalVariable(name: "FuncVar",
