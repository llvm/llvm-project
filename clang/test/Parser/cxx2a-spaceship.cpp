// RUN: %clang_cc1 -std=c++2a -verify %s

template<int> struct X {};

X<1> operator<<(X<0>, X<0>);
X<2> operator<=>(X<0>, X<1>);
X<2> operator<=>(X<1>, X<0>);
X<3> operator<(X<0>, X<2>);
X<3> operator<(X<2>, X<0>);

void f(X<0> x0, X<1> x1) {
  X<2> a = x0 <=> x0 << x0;
  X<2> b = x0 << x0 <=> x0; // expected-warning {{overloaded operator << has higher precedence than comparison operator}} expected-note 2{{}}
  X<3> c = x0 < x0 <=> x1;
  X<3> d = x1 <=> x0 < x0;
  X<3> e = x0 < x0 <=> x0 << x0;
  X<3> f = x0 << x0 <=> x0 < x0; // expected-warning {{overloaded operator << has higher precedence than comparison operator}} expected-note 2{{}}
}
