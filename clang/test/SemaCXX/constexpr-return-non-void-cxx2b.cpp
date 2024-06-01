// RUN: %clang_cc1 -Wno-return-type -std=c++23 -fsyntax-only -verify %s
// expected-no-diagnostics
constexpr int f() { }
static_assert(__is_same(decltype([] constexpr -> int { }( )), int));

consteval int g() { }
static_assert(__is_same(decltype([] consteval -> int { }( )), int));
