// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="remarks=pset" -verify-diagnostics -o %t-out.cir
// XFAIL: *

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
  return p;
}

int *p1(bool b = true) {
  int *p = nullptr;
  if (b) {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid, nullptr }}}
  return p;
}

void p2(int b) {
  int *p = nullptr;
  switch (int x = 0; b) {
  case 1:
    p = &x;
  case 2:
    *p = 42; // expected-warning {{use of invalid pointer 'p'}}
    // expected-remark@-1 {{pset => { nullptr }}}
    break;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { nullptr, invalid }}}
}
