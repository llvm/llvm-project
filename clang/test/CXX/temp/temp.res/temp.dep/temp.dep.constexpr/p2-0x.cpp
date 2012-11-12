// RUN: %clang_cc1 -std=c++11 -verify %s
// expected-no-diagnostics

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
    typename S<k>::T check3a; // ok, u is value-dependent

    constexpr const int *p = &k;
    typename S<*p>::T check3b; // ok, p is value-dependent

    // (missing from the standard)
    //  - a reference and is initialized with an expression that is
    //  value-dependent.
    const int &i = k;
    typename S<i>::T check4; // ok, i is value-dependent
  }
};
