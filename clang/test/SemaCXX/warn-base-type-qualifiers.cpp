// RUN: %clang_cc1 %s -std=c++11 -Wignored-qualifiers -verify

class A { };

typedef const A A_Const;
class B : public A_Const { }; // expected-warning {{'const' qualifier on base class type 'A_Const' (aka 'const A') have no effect}} \
                              // expected-note {{base class 'A_Const' (aka 'const A') specified here}}

typedef const volatile A A_Const_Volatile;
class C : public A_Const_Volatile { }; // expected-warning {{'const volatile' qualifiers on base class type 'A_Const_Volatile' (aka 'const volatile A') have no effect}} \
                                       // expected-note {{base class 'A_Const_Volatile' (aka 'const volatile A') specified here}}

struct D {
  D(int);
};
template <typename T> struct E : T {
  using T::T;
  E(int &) : E(0) {}
};
E<const D> e(1);
