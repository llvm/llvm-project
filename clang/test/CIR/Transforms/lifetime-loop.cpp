// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="history=invalid,null;remarks=pset-invalid" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

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

// p1179r1: 2.4.9.3
void loop0(bool b, int j) {
  int a[4], c[4];
  int *p = &a[0];
  while (j) {
    // This access is invalidated after the first iteration
    *p = 42;     // expected-warning {{use of invalid pointer 'p'}}
                 // expected-remark@-1 {{pset => { c, nullptr }}}
    p = nullptr; // expected-note {{invalidated here}}
    if (b) {
      p = &c[j];
    }
    j = j - 1;
  }
  *p = 0; // expected-warning {{use of invalid pointer 'p'}}
          // expected-remark@-1 {{pset => { a, c, nullptr }}}
}
