// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct X {
  void f() &;
  void g() &&;
};

void (X::*pmf)() & = &X::f;

void fn() {
  void (*[[attr]] fn_ptr)() = &fn; // expected-warning{{unknown attribute 'attr' ignored}}
  void (*[[attrA]] *[[attrB]] fn_ptr_ptr)() = &fn_ptr; // expected-warning{{unknown attribute 'attrA' ignored}} expected-warning{{unknown attribute 'attrB' ignored}}

  void (&[[attr]] fn_lref)() = fn; // expected-warning{{unknown attribute 'attr' ignored}}
  void (&&[[attr]] fn_rref)() = fn; // expected-warning{{unknown attribute 'attr' ignored}}

  int i[5];
  int (*[[attr(i[1])]] pi);  // expected-warning{{unknown attribute 'attr' ignored}}
  pi = &i[0];
}
