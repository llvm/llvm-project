// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

// expected-no-diagnostics

template <decltype(auto) a>
struct S {
    static constexpr int i = 42;
};

template <decltype(auto) a> requires true
struct S<a> {
    static constexpr int i = 0;
};

static constexpr int a = 0;

void test() {
    static_assert(S<a>::i == 0);
    static_assert(S<(a)>::i == 0);
    static_assert(S<((a))>::i == 0);
}
