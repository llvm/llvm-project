// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check -verify-diagnostics -o %t-out.cir
// XFAIL: *

int *basic() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  return p;
}
