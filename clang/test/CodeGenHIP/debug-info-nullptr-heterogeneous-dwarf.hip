// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -emit-llvm -fcuda-is-device -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck %s

// Test that the special case in EmitGlobalVariableForHeterogeneousDwarf
// handles nullptr constants when the are actually folded. This is reduced from
// a case which previously crashed the compiler, and the typedef is seemingly
// required to get Clang to actually fold the constant in this case.

typedef int *ptr;
const ptr constant_nullptr = nullptr;
__attribute__((device)) bool isnull(ptr p) {
  return p == constant_nullptr;
}

// CHECK: !DIGlobalVariableExpression(var: [[VAR:![0-9]+]], expr: !DIExpression(DIOpConstant(i64 0)))
// CHECK: [[VAR]] = distinct !DIGlobalVariable(name: "constant_nullptr"
