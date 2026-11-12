// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// A default template-argument may be specified for any kind of
// template-parameter that is not a template parameter pack.
template<typename ...Types = int> // expected-error{{template parameter pack cannot have a default argument}}
struct X0;

template<int ...Values = 0> // expected-error{{template parameter pack cannot have a default argument}}
struct X1;

template<typename T> struct vector;

template<template<class> class ...Templates = vector> // expected-error{{template parameter pack cannot have a default argument}}
struct X2; 

struct X3 {
  template<typename T = int> // expected-error{{default template argument not permitted on a friend template}}
  friend void f0(X3);

  template<typename T = int>
  friend void f1(X3) {
  }
};

namespace PR8748 {
  // Testcase 1
  struct A0 { template<typename U> struct B; }; 
  template<typename U = int> struct A0::B { };
  
  // Testcase 2
  template<typename T> struct A1 { template<typename U> struct B; }; 
  template<typename T> template<typename U = int> struct A1<T>::B { }; // expected-error{{cannot add a default template argument to the definition of a member of a class template}}

  // Testcase 3
  template<typename T>
  struct X2 {
    void f0();
    template<typename U> void f1();
  };
  
  template<typename T = int> void X2<T>::f0() { } // expected-error{{cannot add a default template argument to the definition of a member of a class template}} 
  template<typename T> template<typename U = int> void X2<T>::f1() { } // expected-error{{cannot add a default template argument to the definition of a member of a class template}}

  namespace Inner {
    template<typename T> struct X3;
    template<typename T> void f2();
  }

  // Okay; not class members.
  template<typename T = int> struct Inner::X3 { };
  template<typename T = int> void Inner::f2() {}
}

namespace PR10069 {
  template<typename T, T a, T b=0, T c=1>
  T f(T x);

  void g() {
    f<unsigned int, 0>(0);
  }
}
