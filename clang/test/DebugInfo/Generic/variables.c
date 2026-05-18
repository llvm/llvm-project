// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s

// CHECK: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+1]]
int global = 42;

// CHECK: DIGlobalVariable({{.*}}line: [[@LINE+4]],{{.*}} type: [[TYPEID:![0-9]+]]
// CHECK: [[TYPEID]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[BASETYPE:![0-9]+]]
// CHECK: [[BASETYPE]] = !DIBasicType(name: "char"
const char* s() {
  return "1234567890";
}

// Note: Windows has `q -> p -> r` ordering and Linux has `p -> q -> r`.
// CHECK-DAG: DILocalVariable(name: "p"
// CHECK-DAG: DILocalVariable(name: "q"
// CHECK-DAG: DILocalVariable(name: "r"
int sum(int p, int q) {
  int r = p + q;
  return r;
}
