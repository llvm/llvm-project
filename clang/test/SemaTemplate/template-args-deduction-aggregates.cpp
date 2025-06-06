// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20

// expected-no-diagnostics

namespace GH113659 {
template <class... Args> struct S {};
struct T {};
struct U {};

template <class... Args> struct B : S<Args...>, Args... {};
B b{S<T, U>{}};
}
