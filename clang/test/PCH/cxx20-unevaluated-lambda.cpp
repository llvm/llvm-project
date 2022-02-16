// RUN: %clang_cc1 -std=c++20 -include %s %s -o %t

// RUN: %clang_cc1 -std=c++20 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++20 -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template<typename T> auto f() -> decltype([]{ return T(42); });

#else /*included pch*/

static_assert(decltype(f<int>())()() == 42);

#endif // HEADER
