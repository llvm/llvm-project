// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s
// expected-no-diagnostics

template<class T> concept IsOk = requires() { typename T::Float; };

template<IsOk T> struct Thing;

template<IsOk T> struct Foobar {
  template<int> struct Inner {
    template<IsOk T2> friend struct Thing;
  };
};

struct MyType { using Float=float; };
Foobar<MyType>::Inner<0> foobar;
