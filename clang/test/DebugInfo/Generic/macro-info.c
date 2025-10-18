// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s

#define GLOBAL(num) global##num
#define DECL_GLOBAL(x) int x
#define SAME_ORDER(x, y) x; y
#define SWAP_ORDER(x,y) y; x

// CHECK: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+4]]
// CHECK: DIGlobalVariable({{.*}}line: [[@LINE+6]],{{.*}} type: [[TYPEID:![0-9]+]]
SAME_ORDER(
  int
    GLOBAL  // <- global
      () = 42,
  const char* s() {
    return "1234567890";
  }
)
// CHECK: DIGlobalVariable(name: "global3",{{.*}} line: [[@LINE+6]]
// CHECK: DIGlobalVariable(name: "global2",{{.*}} line: [[@LINE+2]]
SWAP_ORDER(
  int GLOBAL(  // <- global2
    2) = 43,
  DECL_GLOBAL(
    GLOBAL(  // <- global3
      3)) = 44
);


// CHECK: DIGlobalVariable(name: "global4",{{.*}} line: [[@LINE+2]]
DECL_GLOBAL(
  GLOBAL(  // <- global4
    4));
