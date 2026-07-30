// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

template <typename T>
__global__ T foo() {
  // expected-note@-1 {{kernel function type 'T ()' must have void return type}}
}

void f0() {
  foo<void><<<0, 0>>>();
  foo<int><<<0, 0>>>();
  // expected-error@-1 {{no matching function for call to 'foo'}}
}

__global__ auto f1() {
}

__global__ auto f2(int x) {
  return x + 1;
  // expected-error@-2 {{kernel function type 'auto (int)' must have void return type}}
}

template <bool Cond, typename T = void> struct enable_if { typedef T type; };
template <typename T> struct enable_if<false, T> {};

template <int N>
__global__
auto bar() -> typename enable_if<N == 1>::type {
  // expected-note@-1 {{requirement '3 == 1' was not satisfied [with N = 3]}}
}

template <int N>
__global__
auto bar() -> typename enable_if<N == 2>::type {
  // expected-note@-1 {{requirement '3 == 2' was not satisfied [with N = 3]}}
}

void f3() {
  bar<1><<<0, 0>>>();
  bar<2><<<0, 0>>>();
  bar<3><<<0, 0>>>();
  // expected-error@-1 {{no matching function for call to 'bar'}}
}
