// RUN: %clang_cc1 -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

// RUN: %clang_cc1 -std=c++11 -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<typename T, int N>
using vec = T __attribute__((ext_vector_type(N)));

#else

void test() {
  vec<float, 2> a;  // expected-error@-5 {{zero vector size}}
  vec<float, 0> b; // expected-note {{in instantiation of template type alias 'vec' requested here}}
}

#endif
