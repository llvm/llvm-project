// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple i386-pc-windows-msvc19.16.0 | FileCheck %s
// REQUIRES: asserts, x86-registered-target

// CHECK-LABEL: define {{.*}}@"?f@@YAXXZ"(
// CHECK: call void @"?dc@z@@SAXU1@@Z"

// CHECK-LABEL: define {{.*}}@"?dc@z@@SAXU1@@Z"(
// CHECK: store void ({}*)* %{{.*}}, void ({}*)** %{{.*}}
struct z {
  static void dc(z) {}
  void (*p)(z);
};

void f() {
  z::dc({});
}
