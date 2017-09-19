// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11

// Test reproduces a pair of crashes that were caused by code attempting
// to materialize a default constructor's exception specifier.

template <class T> struct A {
  static T tab[];

  const int M = UNDEFINED; // expected-error {{use of undeclared identifier}}

  int main()
  {
    A<char> a;

    return 0;
  }
};

template <class T> struct B {
  static T tab[];

  // expected-error@+1 {{invalid application of 'sizeof' to an incomplete type}}
  const int N = sizeof(B<char>::tab) / sizeof(char);

  int main()
  {
    B<char> b;

    return 0;
  }
};
