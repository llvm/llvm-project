// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics
namespace A {
template <typename T>
concept C = true;

template <typename T>
requires C<T> && C<T>
void f() {}

template <typename T>
requires C<T> && true
void f() {}

void test() { f<int>(); };
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
template <typename T> concept C5 = true;
template <typename T> concept C6 = true;
template <typename T> concept C7 = true;
template <typename T> concept C8 = true;
template <typename T> concept C9 = true;

template <typename T>
concept Majority8 =
    (C0<T> && C1<T> && C2<T> && C3<T>) ||
    (C0<T> && C1<T> && C2<T> && C4<T>) ||
    (C0<T> && C1<T> && C2<T> && C5<T>) ||
    (C0<T> && C1<T> && C2<T> && C6<T>) ||
    (C0<T> && C1<T> && C2<T> && C7<T>) ||
    (C0<T> && C1<T> && C3<T> && C4<T>) ||
    (C0<T> && C1<T> && C3<T> && C5<T>) ||
    (C0<T> && C1<T> && C3<T> && C6<T>) ||
    (C0<T> && C1<T> && C3<T> && C7<T>) ||
    (C0<T> && C1<T> && C4<T> && C5<T>) ||
    (C0<T> && C1<T> && C4<T> && C6<T>) ||
    (C0<T> && C1<T> && C4<T> && C7<T>) ||
    (C0<T> && C1<T> && C5<T> && C6<T>) ||
    (C0<T> && C1<T> && C5<T> && C7<T>) ||
    (C0<T> && C1<T> && C6<T> && C7<T>) ||
    (C0<T> && C2<T> && C3<T> && C4<T>) ||
    (C0<T> && C2<T> && C3<T> && C5<T>) ||
    (C0<T> && C2<T> && C3<T> && C6<T>) ||
    (C0<T> && C2<T> && C3<T> && C7<T>) ||
    (C0<T> && C2<T> && C4<T> && C5<T>) ||
    (C0<T> && C2<T> && C4<T> && C6<T>) ||
    (C0<T> && C2<T> && C4<T> && C7<T>) ||
    (C0<T> && C2<T> && C5<T> && C6<T>) ||
    (C0<T> && C2<T> && C5<T> && C7<T>) ||
    (C0<T> && C2<T> && C6<T> && C7<T>) ||
    (C0<T> && C3<T> && C4<T> && C5<T>) ||
    (C0<T> && C3<T> && C4<T> && C6<T>) ||
    (C0<T> && C3<T> && C4<T> && C7<T>) ||
    (C0<T> && C3<T> && C5<T> && C6<T>) ||
    (C0<T> && C3<T> && C5<T> && C7<T>) ||
    (C0<T> && C3<T> && C6<T> && C7<T>) ||
    (C0<T> && C4<T> && C5<T> && C6<T>) ||
    (C0<T> && C4<T> && C5<T> && C7<T>) ||
    (C0<T> && C4<T> && C6<T> && C7<T>) ||
    (C0<T> && C5<T> && C6<T> && C7<T>) ||
    (C1<T> && C2<T> && C3<T> && C4<T>) ||
    (C1<T> && C2<T> && C3<T> && C5<T>) ||
    (C1<T> && C2<T> && C3<T> && C6<T>) ||
    (C1<T> && C2<T> && C3<T> && C7<T>) ||
    (C1<T> && C2<T> && C4<T> && C5<T>) ||
    (C1<T> && C2<T> && C4<T> && C6<T>) ||
    (C1<T> && C2<T> && C4<T> && C7<T>) ||
    (C1<T> && C2<T> && C5<T> && C6<T>) ||
    (C1<T> && C2<T> && C5<T> && C7<T>) ||
    (C1<T> && C2<T> && C6<T> && C7<T>) ||
    (C1<T> && C3<T> && C4<T> && C5<T>) ||
    (C1<T> && C3<T> && C4<T> && C6<T>) ||
    (C1<T> && C3<T> && C4<T> && C7<T>) ||
    (C1<T> && C3<T> && C5<T> && C6<T>) ||
    (C1<T> && C3<T> && C5<T> && C7<T>) ||
    (C1<T> && C3<T> && C6<T> && C7<T>) ||
    (C1<T> && C4<T> && C5<T> && C6<T>) ||
    (C1<T> && C4<T> && C5<T> && C7<T>) ||
    (C1<T> && C4<T> && C6<T> && C7<T>) ||
    (C1<T> && C5<T> && C6<T> && C7<T>) ||
    (C2<T> && C3<T> && C4<T> && C5<T>) ||
    (C2<T> && C3<T> && C4<T> && C6<T>) ||
    (C2<T> && C3<T> && C4<T> && C7<T>) ||
    (C2<T> && C3<T> && C5<T> && C6<T>) ||
    (C2<T> && C3<T> && C5<T> && C7<T>) ||
    (C2<T> && C3<T> && C6<T> && C7<T>) ||
    (C2<T> && C4<T> && C5<T> && C6<T>) ||
    (C2<T> && C4<T> && C5<T> && C7<T>) ||
    (C2<T> && C4<T> && C6<T> && C7<T>) ||
    (C2<T> && C5<T> && C6<T> && C7<T>) ||
    (C3<T> && C4<T> && C5<T> && C6<T>) ||
    (C3<T> && C4<T> && C5<T> && C7<T>) ||
    (C3<T> && C4<T> && C6<T> && C7<T>) ||
    (C3<T> && C5<T> && C6<T> && C7<T>) ||
    (C4<T> && C5<T> && C6<T> && C7<T>);

template <typename T>concept Y = C0<T> || Majority8<T>;
template <typename T>concept Z = Majority8<T> && C1<T>;

constexpr int foo(Majority8 auto x) { return 10; }
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
// randomly generated formulas misshandled by clang 20,
// and some other compilers. There is no particular meaning
// to it except to stress-test the compiler.

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

template<typename T>
concept Z10 = true;

template<typename T>
concept Z11 = true;

template<typename T>
concept Z12 = true;

template<typename T>
concept Z13 = true;

template<typename T>
concept Z14 = true;

template<typename T>
concept Z15 = true;

template<typename T>
concept Z16 = true;

template<typename T>
concept Z17 = true;

template<typename T>
concept Z18 = true;

template<typename T>
concept Z19 = true;

namespace T1 {
template<typename T>
concept X = ((((((((Z13<T> || (Z2<T> || Z10<T>)) && (Z2<T> && (Z6<T> && ((Z7<T> && Z13<T>) && Z0<T>)))) && (Z13<T> ||
(Z12<T> && Z8<T>))) && (Z9<T> || (Z2<T> && Z17<T>))) || Z2<T>) && ((((Z17<T> || Z6<T>) && (((Z6<T> || Z4<T>) || Z9<T>)
&& Z13<T>)) || ((Z14<T> || Z10<T>) || Z3<T>)) || (Z8<T> || ((Z19<T> && (Z3<T> && Z14<T>)) && ((Z5<T> || (Z3<T> &&
Z5<T>)) || (Z7<T> && Z13<T>)))))) || ((((Z14<T> && (Z2<T> && Z1<T>)) || ((Z17<T> && Z12<T>) && (Z0<T> || ((((Z9<T> ||
(Z6<T> && Z16<T>)) && Z19<T>) && (Z6<T> && (Z12<T> && Z17<T>))) && (Z19<T> && Z8<T>))))) || (((Z10<T> || Z17<T>) &&
Z1<T>) && ((Z16<T> && (Z15<T> || Z5<T>)) || ((Z4<T> && Z5<T>) || ((Z1<T> || Z4<T>) || Z2<T>))))) && (((Z12<T> && (Z5<T>
&& Z10<T>)) || ((Z4<T> && Z18<T>) && Z0<T>)) || ((((Z10<T> || Z0<T>) && Z18<T>) || (Z15<T> && ((Z11<T> && Z5<T>) &&
Z6<T>))) && Z2<T>)))) && ((((((((((Z8<T> && Z13<T>) && Z7<T>) && Z18<T>) && ((((Z7<T> && Z11<T>) || (Z19<T> && Z6<T>))
|| Z13<T>) && Z15<T>)) || (Z1<T> || Z15<T>)) || (Z9<T> && (Z6<T> || Z10<T>))) || Z0<T>) && (((Z14<T> || Z4<T>) &&
(Z4<T> || ((Z4<T> && Z10<T>) && Z11<T>))) || Z4<T>)) && ((((((Z8<T> && ((Z1<T> && (Z16<T> && (Z0<T> && Z6<T>))) &&
(Z1<T> && Z10<T>))) && ((Z18<T> && Z3<T>) || ((Z14<T> && Z1<T>) || Z15<T>))) && (((Z19<T> || Z17<T>) || ((Z17<T> &&
(Z9<T> && Z19<T>)) || Z6<T>)) || (((Z4<T> || (((Z4<T> || Z9<T>) && Z6<T>) && Z2<T>)) || ((Z17<T> && (Z16<T> && ((Z14<T>
&& Z10<T>) && Z17<T>))) || Z9<T>)) && (Z5<T> && Z6<T>)))) && (((Z3<T> && Z14<T>) || Z5<T>) && Z8<T>)) && ((((Z10<T> ||
(Z17<T> && Z8<T>)) || ((Z16<T> && (((Z12<T> && Z16<T>) || Z18<T>) || (Z4<T> && Z13<T>))) || (Z17<T> && Z10<T>))) ||
((((Z9<T> && ((Z7<T> || Z2<T>) && Z15<T>)) || ((Z18<T> && Z13<T>) || (Z4<T> || Z14<T>))) || (((Z7<T> || (Z10<T> &&
(Z14<T> && Z18<T>))) && (Z9<T> || Z5<T>)) || (Z8<T> && ((Z14<T> || Z11<T>) || ((Z4<T> || Z2<T>) && (Z7<T> &&
Z5<T>)))))) && (((Z14<T> && (Z13<T> && Z10<T>)) || Z8<T>) && (((((Z7<T> || (Z8<T> && Z14<T>)) || Z0<T>) && Z0<T>) ||
Z17<T>) || Z5<T>)))) && (Z16<T> && Z4<T>))) && (((Z1<T> && Z12<T>) || ((Z17<T> || Z4<T>) || (Z15<T> || (Z6<T> ||
Z8<T>)))) || (((Z2<T> || Z19<T>) && Z5<T>) && Z1<T>)))) || ((((Z9<T> || (Z12<T> || Z6<T>)) && (Z5<T> || Z12<T>)) &&
((Z1<T> || Z8<T>) || (Z18<T> && Z19<T>))) || ((Z11<T> && Z17<T>) || (Z5<T> && Z12<T>)))));

template<typename T>
concept Y = Z0<T> && X<T>;

constexpr int foo(X auto x) { return 1; }
constexpr int foo(Y auto y) { return 2; }
static_assert(foo(0) == 2);
}

namespace T3 {

template<typename T>
concept X = (((Z2<T> && ((Z7<T> || (Z8<T> && (Z6<T> && Z4<T>))) && ((Z1<T> && Z3<T>) || ((Z1<T> && (Z7<T> && Z2<T>)) &&
Z1<T>)))) && ((Z7<T> || (((Z6<T> || Z0<T>) || (Z5<T> || Z3<T>)) && Z3<T>)) && ((Z6<T> || ((((Z6<T> && Z8<T>) && (Z8<T>
&& Z3<T>)) || (Z6<T> && Z5<T>)) && (Z6<T> || (Z3<T> && (Z3<T> || Z8<T>))))) && ((((Z3<T> || (Z3<T> && (Z6<T> ||
Z8<T>))) && Z3<T>) && Z9<T>) || ((Z7<T> || Z6<T>) || ((Z3<T> && (Z4<T> && (Z0<T> && Z3<T>))) && (((Z5<T> && (Z1<T> ||
Z5<T>)) || Z3<T>) && (((Z7<T> || Z5<T>) || ((Z9<T> || Z1<T>) && ((Z9<T> && Z0<T>) || Z0<T>))) && (Z5<T> &&
Z7<T>))))))))) || (((((Z5<T> || Z0<T>) || (Z7<T> && (Z8<T> && (Z9<T> || (Z6<T> && Z1<T>))))) || (((Z6<T> || Z3<T>) ||
Z1<T>) && Z3<T>)) || (((Z9<T> && ((((Z9<T> || (Z9<T> && (((Z7<T> || ((Z4<T> || Z3<T>) || Z8<T>)) && Z3<T>) && (Z1<T> &&
Z3<T>)))) || ((Z1<T> && ((Z8<T> && (Z9<T> && Z6<T>)) && (Z1<T> || Z5<T>))) || Z0<T>)) && Z2<T>) && ((Z1<T> || (Z0<T> ||
Z7<T>)) || (Z9<T> && Z4<T>)))) || (Z4<T> || Z3<T>)) && ((Z3<T> && Z9<T>) || ((Z6<T> || Z8<T>) && (Z7<T> && (Z9<T> ||
(Z3<T> || Z7<T>))))))) && (Z2<T> && (Z7<T> || Z3<T>))));

template<typename T>
concept Y = X<T> && ((Z2<T> && (((Z6<T> || Z5<T>) || Z1<T>) && (Z4<T> || ((Z9<T> || (Z2<T> || Z5<T>)) || Z7<T>)))) ||
((((((Z9<T> || (Z1<T> || Z3<T>)) && Z5<T>) || ((Z5<T> || Z0<T>) || (Z2<T> && Z1<T>))) || Z3<T>) || (((Z0<T> && ((Z4<T>
&& (((Z3<T> && Z0<T>) || (Z1<T> || Z5<T>)) || Z6<T>)) || ((Z7<T> || (Z1<T> || Z8<T>)) || Z8<T>))) && ((Z6<T> || (Z6<T>
|| Z9<T>)) && (Z1<T> || Z0<T>))) || (Z5<T> || (((Z8<T> || Z5<T>) && (((((((Z3<T> || Z2<T>) || Z6<T>) || ((Z6<T> ||
Z4<T>) || ((Z1<T> && Z9<T>) || Z8<T>))) || (Z3<T> && (Z9<T> && (Z6<T> || (Z1<T> || Z0<T>))))) && (((Z3<T> && Z5<T>) ||
(Z4<T> || Z2<T>)) && (Z5<T> && (Z6<T> || (Z0<T> || Z1<T>))))) || Z1<T>) || (Z4<T> || (Z1<T> || Z4<T>)))) && Z9<T>))))
&& ((((Z6<T> || (((Z6<T> && (Z3<T> || Z9<T>)) && Z6<T>) && (Z1<T> && Z9<T>))) && ((Z4<T> && (Z4<T> && Z3<T>)) &&
Z4<T>)) && (((((Z1<T> && Z3<T>) && (Z5<T> && Z2<T>)) || (Z1<T> || (Z9<T> || Z1<T>))) && (Z8<T> || Z1<T>)) || ((Z4<T> ||
Z5<T>) && Z3<T>))) && ((((((Z8<T> || Z4<T>) || (Z6<T> && Z3<T>)) || (Z4<T> || Z0<T>)) || Z4<T>) && (Z7<T> || Z5<T>)) &&
((Z8<T> || (Z2<T> && Z1<T>)) && (Z8<T> || Z1<T>))))));

constexpr int foo(X auto x) { return 1; }
constexpr int foo(Y auto y) { return 2; }
static_assert(foo(0) == 2);

}

}
