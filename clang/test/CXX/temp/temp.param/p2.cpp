// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -DCPP11
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s -DCPP17

// There is no semantic difference between class and typename in a
// template-parameter. typename followed by an unqualified-id names a
// template type parameter.
template<class T> struct X;
template<typename T> struct X;

// typename followed by aqualified-id denotes the type in a non-type
// parameter-declaration.
template<typename T, typename T::type Value> struct Y0;
template<typename T, typename X<T>::type Value> struct Y1;

// A storage class shall not be specified in a template-parameter declaration.
template<static int Value> struct Z; //expected-error{{invalid declaration specifier}}
template<typedef int Value> struct Z0; //expected-error{{expected template parameter}} expected-error{{expected identifier}} expected-error{{extraneous 'template<>' in declaration of struct 'Z0'}} expected-note{{did you mean to use 'typename'?}}
template<extern inline int Value> struct Z1; //expected-error2{{invalid declaration specifier}}
template<virtual int Value> struct Z2; //expected-error{{invalid declaration specifier}}
template<explicit int Value> struct Z3; //expected-error{{invalid declaration specifier}}
template<inline int Value> struct Z4; //expected-error{{invalid declaration specifier}}
template<extern int> struct Z5; //expected-error{{invalid declaration specifier}}
template<static int> struct Z6;  //expected-error{{invalid declaration specifier}}
template<explicit int Value> struct Z7; //expected-error{{invalid declaration specifier}}
template<mutable int> struct Z8; //expected-error{{invalid declaration specifier}}

template<const int> struct Z9; // OK
template<volatile int> struct Z10; // OK



#ifdef CPP11
template<thread_local int> struct Z11; //expected-error{{invalid declaration specifier}}
template<constexpr int> struct Z12; //expected-error{{invalid declaration specifier}}

#endif

#ifdef CPP17
template<auto> struct Z13; // OK
#endif

// Make sure that we properly disambiguate non-type template parameters that
// start with 'class'.
class X1 { };
template<class X1 *xptr> struct Y2 { };

// FIXME: add the example from p2
