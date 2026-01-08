// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s

#define GLOBAL(num) global## num
#define DECL_GLOBAL(x) int x
#define SAME_ORDER(x, y) x; y
#define SWAP_ORDER(x,y) y; x



SAME_ORDER(
  int
// CHECK: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+1]]
    GLOBAL  // <- global
      () = 42,
  const char* s() {
// CHECK: DIGlobalVariable({{.*}}line: [[@LINE+1]],{{.*}} type: [[TYPEID:![0-9]+]]
    return "1234567890";
  }
)

SWAP_ORDER(
  int GLOBAL(  // <- global2
    2) = 43,
// CHECK: DIGlobalVariable(name: "global3",{{.*}} line: [[@LINE+3]]
// CHECK: DIGlobalVariable(name: "global2",{{.*}} line: [[@LINE-3]]
  DECL_GLOBAL(
    GLOBAL(  // <- global3
      3)) = 44
);


DECL_GLOBAL(
// CHECK: DIGlobalVariable(name: "global4",{{.*}} line: [[@LINE+1]]
  GLOBAL(  // <- global4
    4));
