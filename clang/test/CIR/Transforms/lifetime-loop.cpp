// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="history=invalid,null remarks=pset" -verify-diagnostics -o %t-out.cir
// XFAIL: *

void loop_basic_for() {
  int *p = nullptr; // expected-note {{invalidated here}}
  for (int i = 0; i < 10; i = i + 1) {
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
           // expected-remark@-1 {{pset => { nullptr, invalid }}}
}

void loop_basic_while() {
  int *p = nullptr; // expected-note {{invalidated here}}
  int i = 0;
  while (i < 10) {
    int x = 0;
    p = &x;
    *p = 42;
    i = i + 1;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
           // expected-remark@-1 {{pset => { nullptr, invalid }}}
}

void loop_basic_dowhile() {
  int *p = nullptr; // expected-note {{invalidated here}}
  int i = 0;
  do {
    int x = 0;
    p = &x;
    *p = 42;
    i = i + 1;
  } while (i < 10); // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42;          // expected-warning {{use of invalid pointer 'p'}}
                    // expected-remark@-1 {{pset => { nullptr, invalid }}}
}
