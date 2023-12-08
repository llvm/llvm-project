// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

// FIXME: These templates should trigger errors in C++11 mode.

template <char const *p>
class A {
  char const *get_p() { return *p; }
};
template <int p>
class B {
  char const *get_p() { return p; }
};

