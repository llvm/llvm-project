// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t -verify %s

// RUN: %clang_cc1 -pedantic -std=c++1y -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -pedantic -std=c++1y -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<typename T> void f(T t) {
  auto a = t.x;
  decltype(auto) b = t.x;
  auto c = (t.x);
  decltype(auto) d = (t.x);
}

#else

struct Z {
  int x : 5; // expected-note {{bit-field}}
};

// expected-error@15 {{non-const reference cannot bind to bit-field 'x'}}
template void f(Z); // expected-note {{in instantiation of}}

#endif
