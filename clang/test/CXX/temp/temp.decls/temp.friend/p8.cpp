// RUN: %clang_cc1 -fsyntax-only -verify %s
template<class T> class A { }; 

class X {
  template<class T> friend class A<T*>; // expected-error{{partial specialization cannot be declared as a friend}}
};
