// RUN: %clang_cc1 -std=c++20 -include %s %s -o %t

// RUN: %clang_cc1 -std=c++20 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++20 -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int g;
struct A { union { int n, m; }; int *p; int A::*q; char buffer[32]; };

template<A a> constexpr const A &get = a;

constexpr const A &v = get<A{}>;
constexpr const A &w = get<A{1, &g, &A::n, "hello"}>;

#else /*included pch*/

template<A a> constexpr const A &get2 = a;

constexpr const A &v2 = get2<A{}>;
constexpr const A &w2 = get2<A{1, &g, &A::n, "hello\0\0\0\0\0"}>;

static_assert(&v == &v2);
static_assert(&w == &w2);

static_assert(&v != &w);
static_assert(&v2 != &w);
static_assert(&v != &w2);
static_assert(&v2 != &w2);

constexpr const A &v3 = get2<A{.n = 0}>;
constexpr const A &x = get2<A{.m = 0}>;

static_assert(&v == &v3);
static_assert(&v != &x);

#endif // HEADER
