// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify
// expected-no-diagnostics

template<typename...>
concept C = true;

template<typename T>
struct S {
    template<typename U>
    static void foo1(U a, auto b);
    static void foo2(T a, C<T> auto b);
    static void foo3(T a, C<decltype(a)> auto b);
    static void foo4(T a, C<decltype(a)> auto b, const C<decltype(b)> auto &&c);
};

using sf1 = decltype(S<int>::foo1(1, 2));
using sf2 = decltype(S<int>::foo2(1, 2));
using sf3 = decltype(S<int>::foo3(1, 2));
using sf4 = decltype(S<int>::foo4(1, 2, 3));


template<typename... T>
struct G {
    static void foo1(auto a, const C<decltype(a)> auto &&... b);
    static void foo2(auto a, const C<decltype(a), T> auto &&... b);
};

using gf1 = decltype(G<int, char>::foo1('a', 1, 2, 3, 4));
using gf2 = decltype(G<int, char>::foo2('a', 1, 2));
