// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir -fcxx-exceptions -fexceptions | FileCheck %s
// XFAIL: *

// CHECK: cir.func

struct E {};
E e;

void throws() { throw e; }

void bar() {
  try {
    throws();
  } catch (E e) {
  }
}
