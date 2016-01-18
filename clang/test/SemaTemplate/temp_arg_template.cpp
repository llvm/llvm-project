// RUN: %clang_cc1 -fsyntax-only -verify %s

template<template<typename T> class X> struct A; // expected-note 2{{previous template template parameter is here}}

template<template<typename T, int I> class X> struct B; // expected-note{{previous template template parameter is here}}

template<template<int I> class X> struct C;  // expected-note{{previous non-type template parameter with type 'int' is here}}

template<class> struct X; // expected-note{{too few template parameters in template template argument}}
template<int N> struct Y; // expected-note{{template parameter has a different kind in template argument}}
template<long N> struct Ylong; // expected-note{{template non-type parameter has a different type 'long' in template argument}}

namespace N {
  template<class> struct Z;
}
template<class, class> struct TooMany; // expected-note{{too many template parameters in template template argument}}


A<X> *a1; 
A<N::Z> *a2;
A< ::N::Z> *a3;

A<Y> *a4; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
A<TooMany> *a5; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
B<X> *a6; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}
C<Y> *a7;
C<Ylong> *a8; // expected-error{{template template argument has different template parameters than its corresponding template template parameter}}

template<typename T> void f(int);

A<f> *a9; // expected-error{{must be a class template}}

// Evil digraph '<:' is parsed as '[', expect error.
A<::N::Z> *a10; // expected-error{{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}

// Do not do a digraph correction here.
A<: :N::Z> *a11;  // expected-error{{expected expression}} \
          expected-error{{C++ requires a type specifier for all declarations}}

// PR7807
namespace N {
  template <typename, typename = int> 
  struct X
  { };

  template <typename ,int> 
  struct Y
  { X<int> const_ref(); };

  template <template<typename,int> class TT, typename T, int N> 
  int operator<<(int, TT<T, N> a) { // expected-note{{candidate template ignored}}
    0 << a.const_ref(); // expected-error{{invalid operands to binary expression ('int' and 'X<int>')}}
  }

  void f0( Y<int,1> y){ 1 << y; } // expected-note{{in instantiation of function template specialization 'N::operator<<<Y, int, 1>' requested here}}
}

// PR12179
template <typename Primitive, template <Primitive...> class F> // expected-warning {{variadic templates are a C++11 extension}}
struct unbox_args {
  typedef typename Primitive::template call<F> x;
};

template <template <typename> class... Templates> // expected-warning {{variadic templates are a C++11 extension}}
struct template_tuple {};
template <typename T>
struct identity {};
template <template <typename> class... Templates> // expected-warning {{variadic templates are a C++11 extension}}
template_tuple<Templates...> f7() {}

void foo() {
  f7<identity>();
}
