// RUN: %clang_cc1 -fsyntax-only -verify %s 
template<typename> struct Y1;
template<typename, int> struct Y2;

template<class T1 = int, // expected-note{{previous default template argument defined here}}
         class T2>  // expected-error{{template parameter missing a default argument}}
  class B1;

template<template<class> class = Y1, // expected-note{{previous default template argument defined here}}
         template<class> class> // expected-error{{template parameter missing a default argument}}
  class B1t;

template<int N = 5,  // expected-note{{previous default template argument defined here}}
         int M>  // expected-error{{template parameter missing a default argument}}
  class B1n;
