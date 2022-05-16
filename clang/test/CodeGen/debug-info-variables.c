// RUN: %clang_cc1 %s -debug-info-kind=standalone -S -emit-llvm -o - | FileCheck %s

// CHECK: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+1]]
int global = 42;

// CHECK: DIGlobalVariable({{.*}}line: [[@LINE+4]],{{.*}} type: [[TYPEID:![0-9]+]]
// CHECK: [[TYPEID]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[BASETYPE:![0-9]+]]
// CHECK: [[BASETYPE]] = !DIBasicType(name: "char"
const char* s() {
  return "1234567890";
}

// CHECK: DILocalVariable(name: "p"
// CHECK: DILocalVariable(name: "q"
// CHECK: DILocalVariable(name: "r"
int sum(int p, int q) {
  int r = p + q;
  return r;
}
