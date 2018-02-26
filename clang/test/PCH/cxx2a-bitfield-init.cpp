// RUN: %clang_cc1 -std=c++2a -include %s -verify %s
// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s -DPCH

#ifndef HEADER
#define HEADER

struct S {
  unsigned int n : 5 = 49; // expected-warning {{changes value}}
};

int a;
template<bool B> struct T {
  int m : B ? 8 : a = 42;
};

#else

// expected-error@-5 {{constant expression}} expected-note@-5 {{cannot modify}}

static_assert(S().n == 17);
static_assert(T<true>().m == 0);
int q = T<false>().m; // expected-note {{instantiation of}}

#endif
