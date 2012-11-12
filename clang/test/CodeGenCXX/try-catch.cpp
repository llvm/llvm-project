// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

struct X { };

const X g();

void f() {
  try {
    throw g();
    // CHECK: @_ZTI1X to i8
  } catch (const X x) {
  }
}
