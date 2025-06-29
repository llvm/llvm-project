// RUN: %clang_cc1 -std=c++20 -verify %s

template <template <class> class>
struct S
{};

template <class T>
concept C1 = requires
{
  typename S<T::template value_types>; // expected-warning {{the use of the keyword template before the qualified name of a class or alias template without a template argument list is deprecated}}
};

template <class T>
requires C1<T>
struct A {};

template <class T>
requires C1<T> && true
struct A<T> {};
