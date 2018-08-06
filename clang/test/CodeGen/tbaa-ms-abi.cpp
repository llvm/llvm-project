// RUN: %clang_cc1 -triple i686-pc-win32 -disable-llvm-passes -emit-llvm -o - -O1 %s | FileCheck %s -check-prefixes=CHECK,OLD-PATH
// RUN: %clang_cc1 -triple i686-pc-win32 -disable-llvm-passes -emit-llvm -new-struct-path-tbaa -o - -O1 %s | FileCheck %s -check-prefixes=CHECK,NEW-PATH
//
// Test that TBAA works in the Microsoft C++ ABI.  We used to error out while
// attempting to mangle RTTI.

struct StructA {
  int a;
};

struct StructB : virtual StructA {
  StructB();
};

StructB::StructB() {
  a = 42;
// CHECK: store i32 42, i32* {{.*}}, !tbaa [[TAG_A_i32:!.*]]
}

// OLD-PATH: [[TYPE_INT:!.*]] = !{!"int", [[TYPE_CHAR:!.*]], i64 0}
// OLD-PATH: [[TYPE_CHAR]] = !{!"omnipotent char", !
// OLD-PATH: [[TAG_A_i32]] = !{[[TYPE_A:!.*]], [[TYPE_INT]], i64 0}
// OLD-PATH: [[TYPE_A]] = !{!"?AUStructA@@", [[TYPE_INT]], i64 0}
// NEW-PATH: [[TYPE_INT:!.*]] = !{[[TYPE_CHAR:!.*]], i64 4, !"int"}
// NEW-PATH: [[TYPE_CHAR]] = !{{{.*}}, i64 1, !"omnipotent char"}
// NEW-PATH: [[TAG_A_i32]] = !{[[TYPE_A:!.*]], [[TYPE_INT]], i64 0, i64 4}
// NEW-PATH: [[TYPE_A]] = !{[[TYPE_CHAR]], i64 4, !"?AUStructA@@", [[TYPE_INT]], i64 0, i64 4}
