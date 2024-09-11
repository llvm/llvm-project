// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

int bar(...);

template <int> struct Int {};

template <class ...T>
consteval auto foo(T... x) -> decltype(bar(T(x)...)) { return 10; }

template <class ...T>
constexpr auto baz(Int<foo<T>(T())>... x) -> int { return 1; }

static_assert(baz<Int<1>, Int<2>, Int<3>>(Int<10>(), Int<10>(), Int<10>()) == 1);
