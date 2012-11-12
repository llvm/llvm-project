// RUN: %clang_cc1 -verify %s -ftemplate-depth 2

template<int N> struct S { };
template<typename T> S<T() + T()> operator+(T, T); // expected-error {{instantiation exceeded maximum depth}} expected-note 3{{while substituting}}
S<0> s;
int k = s + s;
