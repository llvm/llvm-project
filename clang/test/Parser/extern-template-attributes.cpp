// RUN: %clang_cc1 -std=c++17 -verify %s

template <class>
struct S {};

[[deprecated]] extern template struct S<int>; // expected-error {{an attribute list cannot appear here}}
