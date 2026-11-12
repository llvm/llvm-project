// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

template <template <class> class>
struct S
{};

template <class T>
concept C1 = requires
{
  typename S<T::template value_types>;
};

template <class T>
requires C1<T>
struct A {};

template <class T>
requires C1<T> && true
struct A<T> {};
