// RUN: %clang_cc1 -cxx-abi microsoft -triple i686-pc-win32 -disable-llvm-optzns -emit-llvm -o - -O1 %s | FileCheck %s
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

// CHECK: [[TYPE_CHAR:!.*]] = metadata !{metadata !"omnipotent char", metadata
// CHECK: [[TYPE_INT:!.*]] = metadata !{metadata !"int", metadata [[TYPE_CHAR]], i64 0}
// CHECK: [[TAG_A_i32]] = metadata !{metadata [[TYPE_A:!.*]], metadata [[TYPE_INT]], i64 0}
// CHECK: [[TYPE_A]] = metadata !{metadata !"?AUStructA@@", metadata [[TYPE_INT]], i64 0}
