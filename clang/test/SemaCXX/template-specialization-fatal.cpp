// RUN: %clang_cc1 -verify -fsyntax-only %s
// Verify clang doesn't assert()-fail on template specialization happening after
// fatal error.

#include "not_found.h" // expected-error {{'not_found.h' file not found}}

template <class A, class B, class = void>
struct foo {};

template <class A, class B>
struct foo<A, B, decltype(static_cast<void (*)(B)>(0)(static_cast<A (*)()>(0)()))> {};
