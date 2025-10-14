// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s
#define ID(x) x

// CHECK: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+3]]
// CHECK: DIGlobalVariable({{.*}}line: [[@LINE+5]],{{.*}} type: [[TYPEID:![0-9]+]]
ID(
  int global = 42;

  const char* s() {
    return "1234567890";
  }
)

#define SWAP(x,y) y; x

// CHECK: DIGlobalVariable(name: "global3",{{.*}} line: [[@LINE+4]]
// CHECK: DIGlobalVariable(name: "global2",{{.*}} line: [[@LINE+2]]
SWAP(
  int global2 = 43,
  int global3 = 44
);