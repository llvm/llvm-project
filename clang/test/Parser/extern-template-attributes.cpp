// RUN: %clang_cc1 -std=c++17 -fms-extensions -verify %s

template <class>
struct S {};

[[deprecated]] extern template struct S<int>;              // expected-error {{an attribute list cannot appear here}}
__attribute__((deprecated)) extern template struct S<int>; // expected-error {{an attribute list cannot appear here}}
__declspec(deprecated) extern template struct S<int>;      // expected-error {{expected unqualified-id}}
