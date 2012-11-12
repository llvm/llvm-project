// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *

template<typename T> struct Member0 {
  void f(T t) {
    t;
    t.f;
    t->f;
    
    T* tp;
    tp.f;
    tp->f;

    this->f;
    this.f; // expected-error{{member reference base type 'struct Member0 *const' is not a structure or union}}
  }
};
