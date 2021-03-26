// RUN: %clang_cc1 -ast-print -std=c++20 %s -o - | FileCheck %s

template <bool>
struct enable_if {
};

template <__uint128_t x, typename = typename enable_if<x != 0>::type>
void f();

template <__int128_t>
void f();

using T = decltype(f<0>());

// CHECK: using T = decltype(f<0>());
