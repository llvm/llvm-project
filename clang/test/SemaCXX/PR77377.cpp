// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// expected-no-diagnostics

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T, typename U>
concept same_as = is_same_v<T, U>;

template <auto>
struct A {};

template <same_as<int> auto p>
struct A<p> {};

A<0> a;
