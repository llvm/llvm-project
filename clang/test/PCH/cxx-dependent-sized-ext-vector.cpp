// RUN: %clang_cc1 -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s
// RUN: not %clang_cc1 -std=c++11 -include-pch %t -fno-caret-diagnostics %s 2>&1 | FileCheck %s --implicit-check-not='error:'

// RUN: %clang_cc1 -std=c++11 -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<typename T, int N>
// CHECK: :[[#@LINE+1]]:30: error: zero vector size
using vec = T __attribute__((ext_vector_type(N)));

struct S {};

template <class T>
// CHECK: :[[#@LINE+1]]:1: error: invalid vector element type 'S'
using Alias = T __attribute__((ext_vector_type(4)));

template <class T, int N>
// CHECK: :[[#@LINE+1]]:32: error: invalid vector element type 'S'
using Sized = T __attribute__((vector_size(N)));

template <class T> struct Attributed {
  // CHECK: :[[#@LINE+1]]:29: error: invalid vector element type 'S'
  typedef T (__attribute__((ext_vector_type(4))) Type);
};

#else

void test() {
  vec<float, 2> a;  // expected-error@-20 {{zero vector size}}
  vec<float, 0> b; // expected-note {{in instantiation of template type alias 'vec' requested here}}
}

Alias<S> alias; // expected-error@-18 {{invalid vector element type 'S'}} expected-note {{in instantiation of template type alias 'Alias' requested here}}
Sized<S, 16> invalid_sized; // expected-error@-15 {{invalid vector element type 'S'}} expected-note {{in instantiation of template type alias 'Sized' requested here}}
Attributed<S> attributed; // expected-error@-12 {{invalid vector element type 'S'}} expected-note {{in instantiation of template class 'Attributed<S>' requested here}}

#endif
