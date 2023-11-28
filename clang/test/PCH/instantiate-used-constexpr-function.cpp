// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template<typename T> constexpr T f();
constexpr int g() { return f<int>(); } // #1

#else /*included pch*/

template<typename T> constexpr T f() { return 123; }
int k[g()];

#endif // HEADER
