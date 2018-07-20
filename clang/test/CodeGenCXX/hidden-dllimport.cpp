// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -fvisibility-inlines-hidden -o - %s | FileCheck %s

// We used to declare this hidden dllimport, which is contradictory.

// CHECK: declare dllimport void @"?bar@foo@@QEAAXXZ"(%struct.foo*)

struct __attribute__((dllimport)) foo {
  void bar() {}
};
void zed(foo *p) { p->bar(); }
