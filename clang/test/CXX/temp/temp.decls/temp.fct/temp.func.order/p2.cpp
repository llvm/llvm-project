// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

template<typename T>
concept C = sizeof(T) == sizeof(int);

template<auto N>
struct A;

template<C auto N>
struct A<N>;
