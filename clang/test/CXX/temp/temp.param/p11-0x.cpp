// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// If a template-parameter of a class template or alias template has a default
// template-argument, each subsequent template-parameter shall either have a
// default template-argument supplied or be a template parameter pack.
template<typename> struct vector;

template<typename T = int, typename> struct X3t; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}
template<typename T = int, typename> using A3t = int; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}
template<int V = 0, int> struct X3nt; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}
template<int V = 0, int> using A3nt = int; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}
template<template<class> class M = vector, template<class> class> struct X3tt; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}
template<template<class> class M = vector, template<class> class> using A3tt = int; // expected-error{{template parameter missing a default argument}} expected-note{{previous default template argument defined here}}

template<typename T = int, typename ...Types> struct X2t;
template<typename T = int, typename ...Types> using A2t = X2t<T, Types...>;
template<int V = 0, int ...Values> struct X2nt;
template<int V = 0, int ...Values> using A2nt = X2nt<V, Values...>;
template<template<class> class M = vector, template<class> class... Metas>
  struct X2tt;
template<template<class> class M = vector, template<class> class... Metas>
  using A2tt = X2tt<M, Metas...>;

// If a template-parameter of a primary class template or alias template is a
// template parameter pack, it shall be the last template-parameter.
template<typename ...Types, // expected-error{{template parameter pack must be the last template parameter}}
         int After, int After2>
struct X0t;
X0t<int> pr9789();
template<typename ...Types, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
using A0t = int;

template<int ...Values, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
struct X0nt;
template<int ...Values, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
using A0nt = int;

template<template<typename> class ...Templates, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
struct X0tt;
template<template<typename> class ...Templates, // expected-error{{template parameter pack must be the last template parameter}}
         int After>
using A0tt = int;

// [ Note: These are not requirements for function templates or class
// template partial specializations because template arguments can be
// deduced (14.8.2). -- end note]
template<typename... Types> struct X1t;
template<typename ...Types, typename T> struct X1t<T, Types...> { };

template<int... Values> struct X1nt;
template<int ...Values, int V> struct X1nt<V, Values...> { };

template<template<int> class... Meta> struct X1tt;
template<template<int> class... Meta, template<int> class M> 
  struct X1tt<M, Meta...> { };

template<typename ...Types, typename T>
void f1t(X1t<T, Types...>);

template<int ...Values, int V>
void f1nt(X1nt<V, Values...>);

template<template<int> class... Meta, template<int> class M> 
void f1tt(X1tt<M, Meta...>);

namespace DefaultTemplateArgsInFunction {
  template<typename T = int, typename U>  T &f0(U) { T *x = 0; return *x; }

  void test_f0() {
    int &ir0 = f0(3.14159);
    int &ir1 = f0<int>(3.14159);
    float &fr0 = f0<float>(3.14159);
  }

  template<> int &f0(int*);
  template int &f0(double&);
}
