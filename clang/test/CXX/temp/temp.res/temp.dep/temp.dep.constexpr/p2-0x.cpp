// RUN: %clang_cc1 -std=c++11 -verify %s -fexperimental-new-constant-interpreter

template<int n> struct S; // expected-note 3{{here}}

struct LiteralType {
  constexpr LiteralType(int n) : n(n) {}
  int n;
};

template<int n> struct T {
  T() {
    // An identifier is value-dependent if it is:
    //  - a name declared with a dependent type
    S<n> s;
    S<s> check1; // ok, s is value-dependent
    //  - the name of a non-type template parameter
    typename S<n>::T check2; // ok, n is value-dependent
    //  - a potentially-constant variable that is initialized with an
    //    expression that is value-dependent.
    const int k = n;
    typename S<k>::T check3a; // ok, u is value-dependent

    constexpr const int *p = &k;
    typename S<*p>::T check3b; // ok, p is value-dependent

    const int &i = k;
    typename S<i>::T check4; // ok, i is value-dependent

    static const int ki = 42;
    const int &i2 = ki;
    typename S<i2>::T check5; // expected-error {{undefined template}}

    constexpr LiteralType x = n;
    typename S<true ? 1 : x.n>::T check6; // ok, x is value-dependent

    const LiteralType y = n;
    typename S<true ? 2 : y.n>::T check7; // expected-error {{undefined template}}

    constexpr LiteralType z = 42;
    typename S<true ? 3 : z.n>::T check8; // expected-error {{undefined template}}
  }
};
