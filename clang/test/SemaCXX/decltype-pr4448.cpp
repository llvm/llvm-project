// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

template< typename T, T t, decltype(t+2) v >
struct Convoluted {};

int test_array[5];

Convoluted< int *, test_array, nullptr > tarray;
