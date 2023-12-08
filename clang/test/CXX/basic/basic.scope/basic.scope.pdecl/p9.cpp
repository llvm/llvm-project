// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Template type parameters.
typedef unsigned char T;
template<typename T = T> struct X0 { };
template<> struct X0<unsigned char> { static const bool value = true; };
int array0[X0<>::value? 1 : -1];

// Non-type template parameters.
const int N = 17;
template<int N = N> struct X1 { };
template<> struct X1<17> { static const bool value = true; };
int array1[X1<>::value? 1 : -1];

// Template template parameters.
template<template<class> class X0 = X0> struct X2 { };
template<> struct X2<X0> { static const bool value = true; };
int array2[X2<>::value? 1 : -1];
