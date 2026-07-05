// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -verify %s

void test1() {
  struct S { virtual void f(); };
  // expected-warning@-1{{function 'test1()::S::f' has internal linkage but is not defined}}
  S s;
  // expected-note@-1{{used here}}
}

void test2() {
  struct S;
  struct S { virtual void f(); };
  // expected-warning@-1{{function 'test2()::S::f' has internal linkage but is not defined}}
  S s;
  // expected-note@-1{{used here}}
}
