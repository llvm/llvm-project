// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s

struct A {
  virtual int operator-() = 0;
};

void f(A *a) {
  // CHECK: call i32 %
  -*a;
}
