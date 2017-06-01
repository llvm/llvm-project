// RUN: %clang_cc1 -triple i686-pc-win32 -disable-llvm-passes -emit-llvm -o - -O1 %s | FileCheck %s
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

// CHECK: [[TYPE_INT:!.*]] = !{!"int", [[TYPE_CHAR:!.*]], i64 0}
// CHECK: [[TYPE_CHAR]] = !{!"omnipotent char", !
// CHECK: [[TAG_A_i32]] = !{[[TYPE_A:!.*]], [[TYPE_INT]], i64 0}
// CHECK: [[TYPE_A]] = !{!"?AUStructA@@", [[TYPE_INT]], i64 0}
