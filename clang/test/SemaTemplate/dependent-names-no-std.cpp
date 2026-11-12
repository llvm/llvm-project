// RUN: %clang_cc1 -fsyntax-only -verify %s
//
// The whole point of this test is to verify certain diagnostics work in the
// absence of namespace 'std'.

namespace PR10053 {
  namespace ns {
    struct Data {};
  }

  template<typename T> struct A {
    T t;
    A() {
      f(t); // expected-error {{call to function 'f' that is neither visible in the template definition nor found by argument-dependent lookup}}
    }
  };

  void f(ns::Data); // expected-note {{in namespace 'PR10053::ns'}}

  A<ns::Data> a; // expected-note {{in instantiation of member function}}
}
