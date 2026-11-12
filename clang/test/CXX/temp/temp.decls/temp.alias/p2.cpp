// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T> using U = T;

using I = U<U<U<U<int>>>>;
using I = int;

template<typename A, typename B> using Fst = A;
template<typename A, typename B> using Snd = B;

using I = Fst<Snd<char,int>,double>;

namespace StdExample {
  // Prerequisites for example.
  template<class T, class A> struct vector { /* ... */ };


  template<class T> struct Alloc {};
  template<class T> using Vec = vector<T, Alloc<T>>;
  Vec<int> v;

  template<class T>
    void process(Vec<T>& v) // expected-note {{previous definition is here}}
    { /* ... */ }

  template<class T>
    void process(vector<T, Alloc<T>>& w) // expected-error {{redefinition of 'process'}}
    { /* ... */ }

  template<template<class> class TT>
    void f(TT<int>); // expected-note {{candidate template ignored}}

  template<template<class,class> class TT>
    void g(TT<int, Alloc<int>>);

  int h() {
    f(v); // expected-error {{no matching function for call to 'f'}}
    g(v); // OK: TT = vector
  }


  // v's type is same as vector<int, Alloc<int>>.
  using VTest = vector<int, Alloc<int>>;
  using VTest = decltype(v);
}
