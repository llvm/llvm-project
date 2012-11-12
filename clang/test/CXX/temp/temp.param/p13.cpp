// RUN: %clang_cc1 -fsyntax-only -verify %s 
// expected-no-diagnostics

// The scope of atemplate-parameterextends from its point of
// declaration until the end of its template. In particular, a
// template-parameter can be used in the declaration of subsequent
// template-parameters and their default arguments.

template<class T, T* p, class U = T> class X { /* ... */ }; 
// FIXME: template<class T> void f(T* p = new T); 

// Check for bogus template parameter shadow warning.
template<template<class T> class,
         template<class T> class>
  class B1noshadow;
