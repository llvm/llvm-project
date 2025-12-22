// RUN: %clang_cc1 -Wno-error=return-type -emit-llvm -debug-info-kind=standalone -std=c++11 \
// RUN:   -triple thumbv7-apple-ios %s -o - | FileCheck %s

// This forward-declared scoped enum will be created while building its own
// declcontext. Make sure it is only emitted once.

struct A {
  enum class Return;
  Return f1();
};
A::Return* f2() {}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Return",
// CHECK-SAME:             flags: DIFlagFwdDecl,
// CHECK-NOT:              tag: DW_TAG_enumeration_type, name: "Return"
