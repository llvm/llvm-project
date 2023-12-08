// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s 
// expected-no-diagnostics

template<typename> struct Y1;
template<typename, int> struct Y2;

template<class T1, class T2 = int> using B2 = T1;
template<class T1 = int, class T2> using B2 = T1;

template<template<class> class F, template<class> class G = Y1> using B2t = F<G<int>>;
template<template<class> class F = Y2, template<class> class G> using B2t = F<G<int>>;

template<int N, int M = 5> using B2n = Y2<int, N + M>;
template<int N = 5, int M> using B2n = Y2<int, N + M>;
