// RUN: %clang_cc1 -fsyntax-only -verify %s 
// expected-no-diagnostics
template<typename> struct Y1;
template<typename, int> struct Y2;

template<class T1, class T2 = int> class B2; 
template<class T1 = int, class T2> class B2;

template<template<class, int> class, template<class> class = Y1> class B2t;
template<template<class, int> class = Y2, template<class> class> class B2t;

template<int N, int M = 5> class B2n;
template<int N = 5, int M> class B2n;
