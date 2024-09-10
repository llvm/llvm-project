// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

enum {A, S, D, F};
int main() {
    using asdf = decltype(A);
    using enum asdf; // this line causes the crash
    return 0;
}

namespace N1 {
    enum {A, S, D, F};
    constexpr struct T {
    using asdf = decltype(A);
    using enum asdf;
    } t;

    static_assert(t.D == D);
    static_assert(T::S == S);
}

namespace N2 {
    enum {A, S, D, F};
    constexpr struct T {
    struct {
        using asdf = decltype(A);
        using enum asdf;
    } inner;
    } t;

    static_assert(t.inner.D == D);
    static_assert(t.D == D); // expected-error {{no member named 'D' in 'N2::T'}}
}
