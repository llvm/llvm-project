// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="history=invalid,null" -verify-diagnostics -o %t-out.cir
// XFAIL: *

void s0(int b) {
  int *p = nullptr;
  switch (b) {
  default: {
    int x = 0;
    p = &x;
    *p = 42;
  } // expected-note {{pointee 'x' invalidated at end of scope}}
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}

void s1(int b) {
  int *p = nullptr;
  switch (b) {
  default:
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}

void s2(int b) {
  int *p = nullptr;
  switch (int x = 0; b) {
  default:
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}
