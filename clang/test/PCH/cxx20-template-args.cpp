// RUN: %clang_cc1 -std=c++20 -include %s %s -o %t

// RUN: %clang_cc1 -std=c++20 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++20 -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct A { int n; };

template<A a> constexpr const A &get = a;

constexpr const A &v = get<A{}>;

#else /*included pch*/

template<A a> constexpr const A &get2 = a;

constexpr const A &v2 = get2<A{}>;

static_assert(&v == &v2);

#endif // HEADER
