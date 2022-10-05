// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="history=invalid,null" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

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

void s3(int b) {
  int *p = nullptr; // expected-note {{invalidated here}}
  switch (int x = 0; b) {
  case 1:
    p = &x;
  case 2:
    *p = 42; // expected-warning {{use of invalid pointer 'p'}}
    break;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
}
