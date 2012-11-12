// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

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
template<static int Value> struct Z; // FIXME: expect an error

// Make sure that we properly disambiguate non-type template parameters that
// start with 'class'.
class X1 { };
template<class X1 *xptr> struct Y2 { };

// FIXME: add the example from p2
