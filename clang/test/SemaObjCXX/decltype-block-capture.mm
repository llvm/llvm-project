// RUN: %clang_cc1 -fsyntax-only -std=c++17 -fblocks %s
// expected-no-diagnostics
template<typename T, typename U>
inline constexpr bool is_same_as = false;

template<typename T>
inline constexpr bool is_same_as<T, T> = true;

struct X {};

void f() {
    X x;

    void (^b)(void) = ^{
        decltype(auto) y = (x);
        static_assert(is_same_as<decltype(y), decltype((x))>);
    };
}