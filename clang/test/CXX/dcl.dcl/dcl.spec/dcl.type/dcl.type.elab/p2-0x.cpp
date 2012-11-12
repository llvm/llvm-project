// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct A { typedef int type; };
template<typename T> using X = A; // expected-note {{declared here}}
struct X<int>* p2; // expected-error {{elaborated type refers to a type alias template}}


template<typename T> using Id = T; // expected-note {{declared here}}
template<template<typename> class F>
struct Y {
  struct F<int> i; // expected-error {{elaborated type refers to a type alias template}}
  typename F<A>::type j; // ok

  // FIXME: don't produce the diagnostic both for the definition and the instantiation.
  template<typename T> using U = F<char>; // expected-note 2{{declared here}}
  struct Y<F>::template U<char> k; // expected-error 2{{elaborated type refers to a type alias template}}
  typename Y<F>::template U<char> l; // ok
};
template struct Y<Id>; // expected-note {{requested here}}
