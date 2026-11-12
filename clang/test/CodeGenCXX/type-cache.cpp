// RUN: %clang_cc1 -emit-llvm %s -o - -triple i386-pc-windows-msvc19.16.0 | FileCheck %s
// REQUIRES: asserts, x86-registered-target

// CHECK: call ptr @"?f@@YA?AUz@@XZ"()

struct z {
  z (*p)();
};

z f();

void g() {
  f();
}
