// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct A { typedef int type; };
template<typename T> using X = A; // expected-note {{declared here}}
struct X<int>* p2; // expected-error {{alias template 'X' cannot be referenced with the 'struct' specifier}}


template<typename T> using Id = T; // expected-note {{declared here}}
template<template<typename> class F>
struct Y {
  struct F<int> i; // expected-error {{alias template 'Id' cannot be referenced with the 'struct' specifier}}
  typename F<A>::type j; // ok

  // FIXME: don't produce the diagnostic both for the definition and the instantiation.
  template<typename T> using U = F<char>; // expected-note 2{{declared here}}
  struct Y<F>::template U<char> k; // expected-error 2{{alias template 'U' cannot be referenced with the 'struct' specifier}}
  typename Y<F>::template U<char> l; // ok
};
template struct Y<Id>; // expected-note {{requested here}}
