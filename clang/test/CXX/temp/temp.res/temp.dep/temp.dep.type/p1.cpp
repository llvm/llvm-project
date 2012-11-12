// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s 

// Examples from CWG1056.
namespace Example1 {
  template<class T> struct A;
  template<class T> using B = A<T>;

  template<class T> struct A {
    struct C {};
    B<T>::C bc; // ok, B<T> is the current instantiation.
  };

  template<class T> struct A<A<T>> {
    struct C {};
    B<B<T>>::C bc; // ok, B<B<T>> is the current instantiation.
  };

  template<class T> struct A<A<A<T>>> {
    struct C {};
    B<B<T>>::C bc; // expected-error {{missing 'typename'}}
  };
}

namespace Example2 {
  template<class T> struct A {
    void g();
  };
  template<class T> using B = A<T>;
  template<class T> void B<T>::g() {} // ok.
}
