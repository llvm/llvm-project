// RUN: %clang_cc1 -DSETNODEBUG=0 -emit-llvm -std=c++14 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=YESINFO
// RUN: %clang_cc1 -DSETNODEBUG=1 -emit-llvm -std=c++14 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=NOINFO

#if SETNODEBUG
#define NODEBUG __attribute__((nodebug))
#else
#define NODEBUG
#endif

// Const global variable. Use it so it gets emitted.
NODEBUG static const int const_global_int_def = 1;
void func1(int);
void func2() { func1(const_global_int_def); }
// YESINFO-DAG: !DIGlobalVariable(name: "const_global_int_def"
// NOINFO-NOT:  !DIGlobalVariable(name: "const_global_int_def"

// Global variable with a more involved type.
// If the variable has no debug info, the type should not appear either.
struct S1 {
  int a;
  int b;
};
NODEBUG S1 global_struct = { 2, 3 };
// YESINFO-DAG: !DICompositeType({{.*}} name: "S1"
// NOINFO-NOT:  !DICompositeType({{.*}} name: "S1"
// YESINFO-DAG: !DIGlobalVariable(name: "global_struct"
// NOINFO-NOT:  !DIGlobalVariable(name: "global_struct"

// Static data members. Const member needs a use.
// Also the class as a whole needs a use, so that we produce debug info for
// the entire class (iterating over the members, demonstrably skipping those
// with 'nodebug').
struct S2 {
  NODEBUG static int static_member;
  NODEBUG static const int static_const_member = 4;
};
int S2::static_member = 5;
void func3() {
  S2 junk;
  func1(S2::static_const_member);
}
// YESINFO-DAG: !DIGlobalVariable(name: "static_member"
// NOINFO-NOT:  !DIGlobalVariable(name: "static_member"
// YESINFO-DAG: !DIDerivedType({{.*}} name: "static_const_member"
// NOINFO-NOT:  !DIDerivedType({{.*}} name: "static_const_member"

// Function-local static and auto variables.
void func4() {
  NODEBUG static int static_local = 6;
  NODEBUG        int normal_local = 7;
}
// YESINFO-DAG: !DIGlobalVariable(name: "static_local"
// NOINFO-NOT:  !DIGlobalVariable(name: "static_local"
// YESINFO-DAG: !DILocalVariable(name: "normal_local"
// NOINFO-NOT:  !DILocalVariable(name: "normal_local"

template <typename T>
using y NODEBUG = int;
void func5() {
  NODEBUG typedef int x;
  x a;
  y<int> b;
}
// YESINFO-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "x"
// NOINFO-NOT:  !DIDerivedType(tag: DW_TAG_typedef, name: "x"
// YESINFO-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "y<int>"
// NOINFO-NOT:  !DIDerivedType(tag: DW_TAG_typedef, name: "y<int>"
