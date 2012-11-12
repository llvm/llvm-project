// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++11 -S -emit-llvm -o - %s | FileCheck %s

struct A { int a[1]; };
typedef A x[];
int f() {
  x{{{1}}};
  // CHECK: define i32 @_Z1fv
  // CHECK: store i32 1
  // (It's okay if the output changes here, as long as we don't crash.)
}
