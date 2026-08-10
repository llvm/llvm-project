// RUN: %clang_cc1 -emit-llvm %s -o - -triple i386-pc-windows-msvc19.16.0 | FileCheck %s
// REQUIRES: asserts, x86-registered-target

// CHECK: call void @"?dc@z@@SAXU1@@Z"
struct z {
  static void dc(z);
  void (*p)(z);
};

void f() {
  z::dc({});
}
