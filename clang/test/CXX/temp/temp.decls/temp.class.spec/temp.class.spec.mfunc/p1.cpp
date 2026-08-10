// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<typename T, int N>
struct A;

template<typename T>
struct A<T*, 2> {
  A(T);
  ~A();
  
  void f(T*);
  
  operator T*();
  
  static T value;
};

template<class X> void A<X*, 2>::f(X*) { }

template<class X> X A<X*, 2>::value;

template<class X> A<X*, 2>::A(X) { value = 0; }

template<class X> A<X*, 2>::~A() { }

template<class X> A<X*, 2>::operator X*() { return 0; }
