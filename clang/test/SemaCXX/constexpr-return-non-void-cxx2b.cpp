// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

constexpr int f() { } // expected-warning {{non-void function does not return a value}}
static_assert(__is_same(decltype([] constexpr -> int { }( )), int)); // expected-warning {{non-void lambda does not return a value}}

consteval int g() { } // expected-warning {{non-void function does not return a value}}
static_assert(__is_same(decltype([] consteval -> int { }( )), int)); // expected-warning {{non-void lambda does not return a value}}
