// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="history=invalid,null" -verify-diagnostics -o %t-out.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="history=invalid,null" -clangir-verify-diagnostics -emit-cir %s -o %t.cir
// XFAIL: *

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  return p;
}

int *p1(bool b = true) {
  int *p = nullptr; // expected-note {{invalidated here}}
  if (b) {
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  return p;
}

void p2() {
  int *p = nullptr; // expected-note {{invalidated here}}
  *p = 42;          // expected-warning {{use of invalid pointer 'p'}}
}

void p3() {
  int *p;
  p = nullptr; // expected-note {{invalidated here}}
  *p = 42;     // expected-warning {{use of invalid pointer 'p'}}
}

void p4() {
  int *p;  // expected-note {{uninitialized here}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}

void p5() {
  int *p = nullptr;
  {
    int a[10];
    p = &a[0];
  }        // expected-note {{pointee 'a' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}
