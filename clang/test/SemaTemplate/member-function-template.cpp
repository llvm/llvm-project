// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X {
  template<typename T> T& f0(T);
  
  void g0(int i, double d) {
    int &ir = f0(i);
    double &dr = f0(d);
  }
  
  template<typename T> T& f1(T);
  template<typename T, typename U> U& f1(T, U);
  
  void g1(int i, double d) {
    int &ir1 = f1(i);
    int &ir2 = f1(d, i);
    int &ir3 = f1(i, i);
  }
};

void test_X_f0(X x, int i, float f) {
  int &ir = x.f0(i);
  float &fr = x.f0(f);
}

void test_X_f1(X x, int i, float f) {
  int &ir1 = x.f1(i);
  int &ir2 = x.f1(f, i);
  int &ir3 = x.f1(i, i);
}

void test_X_f0_address() {
  int& (X::*pm1)(int) = &X::f0;
  float& (X::*pm2)(float) = &X::f0;
}

void test_X_f1_address() {
  int& (X::*pm1)(int) = &X::f1;
  float& (X::*pm2)(float) = &X::f1;
  int& (X::*pm3)(float, int) = &X::f1;
}

void test_X_f0_explicit(X x, int i, long l) {
  int &ir1 = x.f0<int>(i);
  int &ir2 = x.f0<>(i);
  long &il1 = x.f0<long>(i);
}

// PR4608
class A { template <class x> x a(x z) { return z+y; } int y; };

// PR5419
struct Functor {
  template <typename T>
  bool operator()(const T& v) const {
    return true;
  }
};

void test_Functor(Functor f) {
  f(1);
}

// Instantiation on ->
template<typename T>
struct X1 {
  template<typename U> U& get();
};

template<typename T> struct X2; // expected-note{{here}}

void test_incomplete_access(X1<int> *x1, X2<int> *x2) {
  float &fr = x1->get<float>();
  (void)x2->get<float>(); // expected-error{{implicit instantiation of undefined template}}
}

// Instantiation of template template parameters in a member function
// template.
namespace TTP {
  template<int Dim> struct X {
    template<template<class> class M, class T> void f(const M<T>&);
  };

  template<typename T> struct Y { };

  void test_f(X<3> x, Y<int> y) { x.f(y); }
}

namespace PR7387 {
  template <typename T> struct X {};

  template <typename T1> struct S {
    template <template <typename> class TC> void foo(const TC<T1>& arg);
  };

  template <typename T1> template <template <typename> class TC>
  void S<T1>::foo(const TC<T1>& arg) {}

  void test(const X<int>& x) {
    S<int> s;
    s.foo(x);
  }
}
