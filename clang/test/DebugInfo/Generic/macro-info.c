// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -o - | FileCheck %s --check-prefix=NEW
// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -debug-info-macro-expansion-loc -o - | FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -debug-info-macro-max-depth=15 -o - | FileCheck %s --check-prefix=DEPTH15
// RUN: %clang_cc1 %s -debug-info-kind=standalone -emit-llvm -debug-info-macro-max-depth=14 -o - | FileCheck %s --check-prefix=DEPTH14

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

// Very deep macro argument expansion chain.
#define ID1(x) x
#define ID2(x) ID1(ID1(x))
#define ID4(x) ID2(ID2(x))
#define ID8(x) ID4(ID4(x))
#define ID16(x) ID8(ID8(x))
#define ID32(x) ID16(ID16(x))
#define ID64(x) ID32(ID32(x))
#define ID128(x) ID64(ID64(x))
#define ID256(x) ID128(ID128(x))
#define ID512(x) ID256(ID256(x))

ID8(
// NEW: DIGlobalVariable(name: "global5",{{.*}} line: [[@LINE+4]]
// OLD: DIGlobalVariable(name: "global5",{{.*}} line: [[@LINE-2]]
// DEPTH15: DIGlobalVariable(name: "global5",{{.*}} line: [[@LINE+2]]
// DEPTH14: DIGlobalVariable(name: "global5",{{.*}} line: [[@LINE-4]]
  int global5);

ID512(
// NEW: DIGlobalVariable(name: "global6",{{.*}} line: [[@LINE-1]]
// OLD: DIGlobalVariable(name: "global6",{{.*}} line: [[@LINE-2]]
  int global6);
