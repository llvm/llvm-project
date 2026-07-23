// Regression test for assertion failure when explicit(bool) is used in pre-C++20
// Fixes GitHub issue #152729
// RUN: %clang_cc1 -std=c++98 -verify %s
// RUN: %clang_cc1 -std=c++03 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s

struct S {
  explicit(true) S(int);
  // expected-warning@-1 {{explicit(bool) is a C++20 extension}}
  
  explicit(false) S(float);
  // expected-warning@-1 {{explicit(bool) is a C++20 extension}}
};
