// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <class T> struct eval; // expected-note 3{{template is declared here}}

template <template <class, class...> class TT, class T1, class... Rest> 
struct eval<TT<T1, Rest...>> { };

template <class T1> struct A; 
template <class T1, class T2> struct B; 
template <int N> struct C; 
template <class T1, int N> struct D; 
template <class T1, class T2, int N = 17> struct E;

eval<A<int>> eA;
eval<B<int, float>> eB;
eval<C<17>> eC; // expected-error{{implicit instantiation of undefined template 'eval<C<17>>'}}
eval<D<int, 17>> eD; // expected-error{{implicit instantiation of undefined template 'eval<D<int, 17>>'}}
eval<E<int, float>> eE; // expected-error{{implicit instantiation of undefined template 'eval<E<int, float>>}}

template<template <int ...N> class TT> struct X0 { }; // expected-note{{previous non-type template parameter with type 'int' is here}}
template<int I, int J, int ...Rest> struct X0a;
template<int ...Rest> struct X0b;
template<int I, long J> struct X0c; // expected-note{{template non-type parameter has a different type 'long' in template argument}}

X0<X0a> inst_x0a;
X0<X0b> inst_x0b;
X0<X0c> inst_x0c; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}

template<typename T, 
         template <T ...N> class TT>  // expected-note{{previous non-type template parameter with type 'short' is here}}
struct X1 { };
template<int I, int J, int ...Rest> struct X1a;
template<long I, long ...Rest> struct X1b;
template<short I, short J> struct X1c;
template<short I, long J> struct X1d; // expected-note{{template non-type parameter has a different type 'long' in template argument}}

X1<int, X1a> inst_x1a;
X1<long, X1b> inst_x1b;
X1<short, X1c> inst_x1c;
X1<short, X1d> inst_x1d; // expected-error{{template template argument has different template parameters than its corresponding template template paramete}}

template <int> class X2; // expected-note{{template is declared here}} \
                         // expected-note{{template is declared here}}
class X3 : X2<1> {}; // expected-error{{implicit instantiation of undefined template 'X2<1>'}}

template <int> class X4 : X3 {
  struct {
    X2<1> e; // expected-error{{implicit instantiation of undefined template 'X2<1>'}}
  } f;
};
