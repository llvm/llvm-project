// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

struct X { };

const X g();

void f() {
  try {
    throw g();
    // CHECK: @_ZTI1X to i8
  } catch (const X x) {
  }
}

void h() {
  try {
    throw "ABC";
    // CHECK: @_ZTIPKc to i8
  } catch (char const(&)[4]) {
  }
}
