// RUN: %clang_cc1 -std=c++98 -verify=cxx98 %s
// RUN: %clang_cc1 -std=c++11 -verify=cxx11 %s
// cxx11-no-diagnostics

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
    typename S<k>::T check3; // ok, k is value-dependent

    const int &i = k; // cxx98-note {{declared here}}
    typename S<i>::T check4; // cxx98-error {{not an integral constant expression}} cxx98-note {{read of variable 'i' of non-integral, non-enumeration type 'const int &'}}
  }
};
