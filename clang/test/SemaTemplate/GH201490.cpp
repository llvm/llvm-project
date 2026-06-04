// RUN: %clang_cc1 -fsyntax-only -verify %s

// regression test for https://github.com/llvm/llvm-project/issues/201490
template<class T> struct A {};
template<class T> struct B : A<T> {};
template<> template<class T> class A<int>::B {}; // expected-error{{out-of-line definition of 'B' does not match any declaration in 'A<int>'}}

// A legitimate member class template explicit specialization
template<class T> struct C { template<class U> struct D; };
  template<> template<class U> struct C<int>::D {};
