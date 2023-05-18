// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> class A {
  typedef int Member;
};

class B {
  template <class T> friend class A;
  template <class T> friend class Undeclared;
  
  template <class T> friend typename A<T>::Member; // expected-error {{friend type templates must use an elaborated type}}
};
