// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s -Wc++98-c++11-c++14-compat

// Check that we don't allow illegal uses of inline
// (checking C++-only constructs here)
struct c {inline int a;}; // expected-error{{'inline' can only appear on functions}}

void localVar() {
  inline int a; // expected-error{{inline declaration of 'a' not allowed in block scope}}
}

// Check that we warn appropriately.
#if __cplusplus <= 201402L
inline int a; // expected-warning{{inline variables are a C++17 extension}}
#else
inline int a; // expected-warning{{inline variables are incompatible with C++ standards before C++17}}
#endif
