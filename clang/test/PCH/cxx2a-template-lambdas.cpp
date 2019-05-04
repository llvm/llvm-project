// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

auto l1 = []<int I>() constexpr -> int {
    return I;
};

auto l2 = []<auto I>() constexpr -> decltype(I) {
    return I;
};

auto l3 = []<class T>(auto i) constexpr -> T {
  return T(i);
};

auto l4 = []<template<class> class T, class U>(T<U>, auto i) constexpr -> U {
  return U(i);
};

#else /*included pch*/

static_assert(l1.operator()<5>() == 5);
static_assert(l1.operator()<6>() == 6);

static_assert(l2.operator()<7>() == 7);
static_assert(l2.operator()<nullptr>() == nullptr);

static_assert(l3.operator()<int>(8.4) == 8);
static_assert(l3.operator()<int>(9.9) == 9);

template<typename T>
struct DummyTemplate { };

static_assert(l4(DummyTemplate<float>(), 12) == 12.0);
static_assert(l4(DummyTemplate<int>(), 19.8) == 19);

#endif // HEADER
