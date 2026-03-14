// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Check for template type parameter pack (mis-)matches with template
// type parameters.
template<typename ...T> struct X0t;
template<typename ...T> struct X0t;

template<typename ...T> struct X1t; // expected-note{{previous template type parameter pack declared here}}
template<typename T> struct X1t; // expected-error{{template type parameter conflicts with previous template type parameter pack}}

template<typename T> struct X2t; // expected-note{{previous template type parameter declared here}}
template<typename ...T> struct X2t; // expected-error{{template type parameter pack conflicts with previous template type parameter}}

template<template<typename ...T> class> struct X0t_intt; 
template<template<typename ...T> class> struct X0t_intt; 

template<template<typename ...T> class> struct X1t_intt; // expected-note{{previous template type parameter pack declared here}}
template<template<typename T> class> struct X1t_intt; // expected-error{{template type parameter conflicts with previous template type parameter pack}}

template<template<typename T> class> struct X2t_intt; // expected-note{{previous template type parameter declared here}}
template<template<typename ...T> class> struct X2t_intt; // expected-error{{template type parameter pack conflicts with previous template type parameter}}

template<int ...Values> struct X1nt; // expected-note{{previous non-type template parameter pack declared here}}
template<int Values> struct X1nt; // expected-error{{non-type template parameter conflicts with previous non-type template parameter pack}}

template<template<class T> class> class X1tt; // expected-note{{previous template template parameter declared here}}
template<template<class T> class...> class X1tt; // expected-error{{template template parameter pack conflicts with previous template template parameter}}

// Check for matching with out-of-line definitions
namespace rdar8859985 {
  template<typename ...> struct tuple { };
  template<int ...> struct int_tuple { };

  template<typename T>
  struct X {
    template<typename ...Args1, int ...Indices1>
    X(tuple<Args1...>, int_tuple<Indices1...>);
  };

  template<typename T>
  template<typename ...Args1, int ...Indices1>
  X<T>::X(tuple<Args1...>, int_tuple<Indices1...>) {}
}
