// RUN: %clang_cc1 -std=c++1y -verify %s
// expected-no-diagnostics

template<typename T> struct A {
  template<typename U> struct B;
  template<typename U> struct B<U*>;
};
template<typename T> template<typename U> struct A<T>::B<U*> {};
template struct A<int>;
A<int>::B<int*> b;


template<typename T> struct B {
  template<typename U> static const int var1;
  template<typename U> static const int var1<U*>;

  template<typename U> static const int var2;
};
template<typename T> template<typename U> const int B<T>::var1<U*> = 1;
template<typename T> template<typename U> const int B<T>::var2<U*> = 1;
template struct B<int>;
int b_test1[B<int>::var1<int*>];
int b_test2[B<int>::var2<int*>];
