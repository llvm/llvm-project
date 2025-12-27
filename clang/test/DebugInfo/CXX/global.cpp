// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Multiple references to the same constant should result in only one entry in
// the globals list.

namespace ns {
const int cnst = 42;
}
int f1() {
  return ns::cnst + ns::cnst;
}

// CHECK: !DICompileUnit(
// CHECK-SAME:           globals: [[GLOBALS:![0-9]*]]

// CHECK: [[GLOBALS]] = !{[[CNST:![0-9]*]]}

// CHECK: [[CNST]] = !DIGlobalVariableExpression(var: [[CNSTV:.*]], expr:
// CHECK: [[CNSTV]] = distinct !DIGlobalVariable(name: "cnst",
// CHECK-SAME:                                   scope: [[NS:![0-9]*]]
// CHECK: [[NS]] = !DINamespace(name: "ns"

