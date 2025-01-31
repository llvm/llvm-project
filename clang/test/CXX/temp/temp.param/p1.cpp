// Suppress 'no run line' failure.
// RUN: %clang_cc1 -fsyntax-only -verify %s

template<template<> class C> class D; // expected-error{{template template parameter must have its own template parameters}}


struct A {};
template<class M,
         class T // expected-note {{template parameter is declared here}}
           = A,  // expected-note{{previous default template argument defined here}}
         class C> // expected-error{{template parameter missing a default argument}}
class X0 {};
X0<int> x0; // expected-error{{missing template argument for template parameter}}
