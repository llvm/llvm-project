// RUN: %clang_cc1 -std=c++98 -verify %s

template<int n> struct S;

template<int n> struct T {
  T() {
    // An identifier is value-dependent if it is:
    //  - a name declared with a dependent type
    S<n> s;
    S<s> check1; // ok, s is value-dependent
    //  - the name of a non-type template parameter
    typename S<n>::T check2; // ok, n is value-dependent
    //  - a constant with literal type and is initialized with an expression
    //  that is value-dependent.
    const int k = n;
    typename S<k>::T check3; // ok, u is value-dependent

    const int &i = k;
    typename S<i>::T check4; // expected-error {{not an integral constant expression}} expected-error {{qualified name}}
  }
};
