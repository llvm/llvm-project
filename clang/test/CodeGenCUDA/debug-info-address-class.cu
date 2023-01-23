// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx-unknown-unknown -debug-info-kind=limited -dwarf-version=2 -debugger-tuning=gdb | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-DAG: ![[FILEVAR0:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_global)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR0]], expr: !DIExpression())
__device__ int FileVar0;
// CHECK-DAG: ![[FILEVAR1:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR1]], expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))
__device__ __shared__ int FileVar1;
// CHECK-DAG: ![[FILEVAR2:[0-9]+]] = distinct !DIGlobalVariable(name: "FileVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: false, isDefinition: true, memorySpace: DW_MSPACE_LLVM_constant)
// CHECK-DAG: !DIGlobalVariableExpression(var: ![[FILEVAR2]], expr: !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef))
__device__ __constant__ int FileVar2;

__device__ void kernel1(
    // CHECK-DAG: ![[ARG:[0-9]+]] = !DILocalVariable(name: "Arg", arg: {{[0-9]+}}, scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
    // CHECK-DAG: call void @llvm.dbg.declare(metadata ptr {{.*}}, metadata ![[ARG]], metadata !DIExpression()), !dbg !{{[0-9]+}}
    int Arg) {
    // CHECK-DAG: ![[FUNCVAR0:[0-9]+]] = distinct !DIGlobalVariable(name: "FuncVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, isLocal: true, isDefinition: true, memorySpace: DW_MSPACE_LLVM_group)
    // CHECK-DAG: !DIGlobalVariableExpression(var: ![[FUNCVAR0]], expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))
  __shared__ int FuncVar0;
  // CHECK-DAG: ![[FUNCVAR1:[0-9]+]] = !DILocalVariable(name: "FuncVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
  // CHECK-DAG: call void @llvm.dbg.declare(metadata ptr {{.*}}, metadata ![[FUNCVAR1]], metadata !DIExpression()), !dbg !{{[0-9]+}}
  int FuncVar1;
}
