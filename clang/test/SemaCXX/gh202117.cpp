// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<template<decltype(foo())> typename T> struct S {}; // expected-error {{use of undeclared identifier 'foo'}}
template<int*> struct P;

S<P> s;
