// RUN: %clang_cc1 -std=c++1z -verify %s

namespace A {
  int m, n;
};

namespace B {
  using A::m, A::n, A::n;
  int q = m + n;
}

struct X {
  int x1, x2, y, z; // expected-note 2{{conflicting}}
};
struct Y {
  int x1, x2, y, z; // expected-note 2{{target}}
};
struct Z : X, Y {
  using X::x1,
        blah::blah, // expected-error {{undeclared}}
        X::x2, // expected-note {{previous}}
        Y::y,
        X::x2, // expected-error {{redeclaration}}
        X::z,
        Y::z; // expected-error {{conflicts with}}
};
int X::*px1 = &Z::x1;
int X::*px2 = &Z::x2;
int Y::*py = &Z::y;
int X::*pz = &Z::z;

template<typename ...T> struct Q : T... {
  using T::z...; // expected-error {{conflicts}}
};
Q<X,Y> q; // expected-note {{instantiation of}}
