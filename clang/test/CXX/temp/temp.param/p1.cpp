// Suppress 'no run line' failure.
// RUN: %clang_cc1 -fsyntax-only -verify %s

template<template<> class C> class D; // expected-error{{template template parameter must have its own template parameters}}


struct A {};
template<class M, 
         class T = A,  // expected-note{{previous default template argument defined here}}
         class C> // expected-error{{template parameter missing a default argument}}
class X0 {}; // expected-note{{template is declared here}}
X0<int> x0; // expected-error{{too few template arguments for class template 'X0'}}
