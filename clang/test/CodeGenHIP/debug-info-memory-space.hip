// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s
// CHECK-DAG: !DIGlobalVariable(name: "GlobalShared", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
// CHECK-DAG: !DIGlobalVariable(name: "GlobalDevice", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: !DIGlobalVariable(name: "GlobalConstant", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: !DIGlobalVariable(name: "FuncVarShared", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
// CHECK-DAG: !DILocalVariable(name: "FuncVar", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})

// CHECK-DAG: !DILocalVariable(name: "FuncVarSharedPointer", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DEVICE_PTR:[0-9]+]])
// CHECK-DAG: !DILocalVariable(name: "FuncVarPointer", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DEVICE_PTR:[0-9]+]])
// CHECK-DAG: ![[DEVICE_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}})

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

__shared__ int GlobalShared;
__device__ int GlobalDevice;
__constant__ int GlobalConstant;

__device__ void kernel1(int Arg) {
  __shared__ int FuncVarShared;
  int FuncVar;

  auto *FuncVarSharedPointer = &FuncVarShared;
  auto *FuncVarPointer = &FuncVar;
}
