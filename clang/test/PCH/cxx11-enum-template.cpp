// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

template<typename T> struct S {
  enum class E {
    e = T()
  };
};

S<int> a;
S<long>::E b;
S<double>::E c;
template struct S<char>;

#else

int k1 = (int)S<int>::E::e;
int k2 = (int)decltype(b)::e;
int k3 = (int)decltype(c)::e; // expected-error@10 {{conversion from 'double' to 'int'}} expected-note {{here}}
int k4 = (int)S<char>::E::e;

#endif
