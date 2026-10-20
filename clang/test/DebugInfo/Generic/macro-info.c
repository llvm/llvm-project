// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s --check-prefix=NEW
// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -debug-info-macro-expansion-loc -o - | FileCheck %s --check-prefix=OLD

#define GLOBAL(num) global## num
#define DECL_GLOBAL(x) int x
#define SAME_ORDER(x, y) x; y
#define SWAP_ORDER(x,y) y; x



SAME_ORDER(
  int
// NEW: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE+2]]
// OLD: DIGlobalVariable(name: "global",{{.*}} line: [[@LINE-3]]
    GLOBAL  // <- global
      () = 42,
  const char* s() {
// NEW: DIGlobalVariable({{.*}}line: [[@LINE+2]],{{.*}} type: [[TYPEID:![0-9]+]]
// OLD: DIGlobalVariable({{.*}}line: [[@LINE-8]],{{.*}} type: [[TYPEID:![0-9]+]]
    return "1234567890";
  }
)

SWAP_ORDER(
  int GLOBAL(  // <- global2
    2) = 43,
// NEW: DIGlobalVariable(name: "global3",{{.*}} line: [[@LINE+5]]
// NEW: DIGlobalVariable(name: "global2",{{.*}} line: [[@LINE-3]]
// OLD: DIGlobalVariable(name: "global3",{{.*}} line: [[@LINE-5]]
// OLD: DIGlobalVariable(name: "global2",{{.*}} line: [[@LINE-6]]
  DECL_GLOBAL(
    GLOBAL(  // <- global3
      3)) = 44
);


DECL_GLOBAL(
// NEW: DIGlobalVariable(name: "global4",{{.*}} line: [[@LINE+2]]
// OLD: DIGlobalVariable(name: "global4",{{.*}} line: [[@LINE-2]]
  GLOBAL(  // <- global4
    4));
