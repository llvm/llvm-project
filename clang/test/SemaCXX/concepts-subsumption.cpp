// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace A {
template <typename T>
concept C = true;

template <typename T>
requires C<T> && C<T>
void f() {}

template <typename T>
requires C<T> && true
void f() {}

template <>
void f<int>();
}

namespace B {
template <typename T>
concept A = true;
template <typename T>
concept B = true;

template <typename T>
requires (A<T> && B<T>)
constexpr int f() { return 0; }

template <typename T>
requires (A<T> || B<T>)
constexpr int f() { return 1; }

static_assert(f<int>() == 0);
}

namespace GH122581 {
// Test that producing a Conjunctive Normal Form
// does not blow up exponentially.
// i.e, this should terminate reasonably quickly
// within a small memory footprint
template <typename T> concept C0 = true;
template <typename T> concept C1 = true;
template <typename T> concept C2 = true;
template <typename T> concept C3 = true;
template <typename T> concept C4 = true;

template <typename T>
concept majority5 =
    (C0<T> && C1<T> && C2<T>) ||
    (C0<T> && C1<T> && C3<T>) ||
    (C0<T> && C1<T> && C4<T>) ||
    (C0<T> && C2<T> && C3<T>) ||
    (C0<T> && C2<T> && C4<T>) ||
    (C0<T> && C3<T> && C4<T>) ||
    (C1<T> && C2<T> && C3<T>) ||
    (C1<T> && C2<T> && C4<T>) ||
    (C1<T> && C3<T> && C4<T>) ||
    (C2<T> && C3<T> && C4<T>);

template <typename T>concept Y = C0<T> && majority5<T>;
template <typename T>concept Z = Y<T> && C1<T>;

constexpr int foo(majority5 auto x) { return 10; }
constexpr int foo(Y auto y) { return 20; }
constexpr int foo(Z auto y) { return 30; }
static_assert(foo(0) == 30);
}

namespace WhateverThisIs {
template <typename T> concept C0 = true;
template <typename T> concept C1 = true;
template <typename T> concept C2 = true;
template <typename T> concept C3 = true;
template <typename T> concept C4 = true;

template <typename T>
concept X =
    (C0<T> || C1<T> || C2<T>) &&
    (C0<T> || C1<T> || C3<T>) &&
    (C0<T> || C1<T> || C4<T>) &&
    (C0<T> || C2<T> || C3<T>) &&
    (C0<T> || C2<T> || C4<T>) &&
    (C0<T> || C3<T> || C4<T>) &&
    (C1<T> || C2<T> || C3<T>) &&
    (C1<T> || C2<T> || C4<T>) &&
    (C1<T> || C3<T> || C4<T>) &&
    (C2<T> || C3<T> || C4<T>);

template <typename T>concept Y = C0<T> && X<T>;

template <typename T>concept Z = Y<T> && C1<T>;

constexpr int foo(X auto x) { return 10; }
constexpr int foo(Y auto y) { return 20; }
constexpr int foo(Z auto y) { return 30; }

static_assert(foo(0) == 30);
}

namespace WAT{
template<typename T>
concept Z0 = true;

template<typename T>
concept Z1 = true;

template<typename T>
concept Z2 = true;

template<typename T>
concept Z3 = true;

template<typename T>
concept Z4 = true;

template<typename T>
concept Z5 = true;

template<typename T>
concept Z6 = true;

template<typename T>
concept Z7 = true;

template<typename T>
concept Z8 = true;

template<typename T>
concept Z9 = true;

template <typename T>
concept X =
    ((((((Z6<T> || Z2<T>)&&Z3<T>)&&(Z2<T> || ((Z2<T> || Z3<T>)&&Z8<T>))) ||
        ((Z0<T> || Z9<T>) ||
        (Z7<T> || ((Z3<T> || (Z9<T> && Z4<T>)) || Z2<T>)))) ||
        ((Z9<T> && (Z7<T> && (Z3<T> || Z2<T>))) ||
        (((Z7<T> || (Z3<T> || Z2<T>)) && ((Z0<T> && Z9<T>)&&(Z1<T> || Z0<T>))) ||
        (Z5<T> || (Z1<T> && Z8<T>))))) &&
        (((((((Z8<T> || (((Z5<T> && Z6<T>)&&(Z4<T> || Z2<T>)) ||
                        (Z8<T> && (Z8<T> && Z7<T>)))) ||
            ((Z1<T> && (Z2<T> || Z6<T>)) || Z4<T>)) &&
            Z8<T>)&&((((((Z6<T> && (Z4<T> || Z3<T>)) ||
                        Z1<T>)&&(Z0<T> && ((Z9<T> && Z6<T>) || Z4<T>))) &&
                        Z0<T>) ||
                    ((Z5<T> || Z8<T>)&&((Z7<T> || (Z6<T> && Z5<T>)) ||
                                        (Z2<T> || (Z3<T> && Z9<T>))))) &&
                    ((((Z2<T> || (Z7<T> || (Z4<T> || Z3<T>))) ||
                        (Z0<T> && Z3<T>)) ||
                        (Z7<T> ||
                        ((Z3<T> || (Z3<T> && (Z3<T> || Z0<T>))) && Z6<T>))) &&
                    ((Z9<T> && Z7<T>) ||
                        (Z4<T> && (((Z7<T> || Z1<T>) || Z4<T>) || Z3<T>)))))) &&
        (((Z1<T> || (((Z0<T> && (Z1<T> || Z5<T>)) && (Z2<T> || Z1<T>)) ||
                        (Z0<T> && ((Z0<T> && Z7<T>)&&Z5<T>)))) &&
            (Z3<T> && (Z1<T> && (Z9<T> && Z1<T>)))) &&
            (((((Z4<T> && Z3<T>) ||
                Z0<T>)&&((Z3<T> || (Z4<T> || Z1<T>)) && Z6<T>)) &&
            Z4<T>)&&((((Z6<T> || ((Z3<T> || Z4<T>) || Z7<T>)) ||
                        Z0<T>)&&Z2<T>) ||
                    Z0<T>)))) ||
        (((Z5<T> && ((Z0<T> || Z5<T>)&&(Z3<T> && ((Z1<T> && Z9<T>)&&Z4<T>)))) ||
            (Z2<T> || Z8<T>)) ||
        (((Z6<T> || Z1<T>) || ((Z6<T> && ((Z6<T> || Z9<T>) || Z8<T>)) ||
                                ((Z7<T> || Z6<T>)&&(Z7<T> || Z4<T>)))) &&
            ((((((Z4<T> && Z8<T>) || Z1<T>) ||
                (((Z6<T> || Z3<T>)&&(Z9<T> || Z1<T>)) ||
                (Z1<T> && (Z0<T> || Z4<T>)))) ||
            (((Z8<T> && (Z1<T> && Z0<T>)) || Z3<T>) ||
                (((((Z5<T> && Z1<T>)&&((Z6<T> && Z7<T>)&&(Z0<T> || Z5<T>))) ||
                (Z0<T> && ((Z2<T> || Z1<T>)&&Z3<T>))) ||
                (Z6<T> && (Z2<T> || Z4<T>))) ||
                Z5<T>))) ||
            (((Z9<T> && Z1<T>) || (Z4<T> && Z6<T>)) &&
            (((Z6<T> || Z1<T>) || ((Z4<T> && Z2<T>) || Z5<T>)) ||
                ((Z5<T> && Z9<T>)&&Z3<T>)))) &&
            (((Z5<T> && (Z7<T> || Z3<T>)) && Z9<T>) || (Z0<T> || Z4<T>)))))) ||
        ((((Z8<T> && (Z1<T> || Z9<T>)) &&
            ((Z4<T> && Z2<T>) ||
            ((Z2<T> || ((Z9<T> && Z7<T>) || Z0<T>)) && Z2<T>))) &&
        (((Z0<T> && Z9<T>)&&Z9<T>)&&(Z6<T> && Z5<T>))) &&
        (((((Z7<T> || Z5<T>) || Z6<T>) ||
            ((Z7<T> || (Z0<T> && Z8<T>)) ||
            ((Z5<T> && (Z3<T> && Z1<T>)) && Z5<T>))) ||
            Z0<T>)&&((Z9<T> && Z7<T>)&&Z4<T>)))));

template<typename T>
concept Y = Z0<T> && X<T>;

constexpr int foo(X auto x) { return 1; } // expected-note{{candidate function}}

constexpr int foo(Y auto y) { return 2; } // expected-note{{candidate function}}

static_assert(foo(0) == 3); // expected-error {{call to 'foo' is ambiguous}}
}
