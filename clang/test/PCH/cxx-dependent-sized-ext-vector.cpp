// RUN: %clang_cc1 -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

// RUN: %clang_cc1 -std=c++11 -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<typename T, int N>
using vec = T __attribute__((ext_vector_type(N)));

struct S {};

template <class T>
using Alias = T __attribute__((ext_vector_type(4)));

template <class T, int N>
using Sized = T __attribute__((vector_size(N)));

template <class T> struct Attributed {
  typedef T (__attribute__((ext_vector_type(4))) Type);
};

#else

void test() {
  vec<float, 2> a;  // expected-error@-17 {{zero vector size}}
  vec<float, 0> b; // expected-note {{in instantiation of template type alias 'vec' requested here}}
}

Alias<S> alias; // expected-error@-16 {{invalid vector element type 'S'}} expected-note {{in instantiation of template type alias 'Alias' requested here}}
Sized<S, 16> invalid_sized; // expected-error@-14 {{invalid vector element type 'S'}} expected-note {{in instantiation of template type alias 'Sized' requested here}}
Attributed<S> attributed; // expected-error@-12 {{invalid vector element type 'S'}} expected-note {{in instantiation of template class 'Attributed<S>' requested here}}

#endif
