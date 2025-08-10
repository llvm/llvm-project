// Regression test for assertion failure when explicit(bool) is used in pre-C++17
// This test ensures no crash occurs and appropriate error messages are shown.
// RUN: %clang_cc1 -std=c++03 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify %s

struct S {
  // Before the fix, this would cause assertion failure in BuildConvertedConstantExpression
  // Now it should produce a proper error message in C++14 and earlier modes
  // Note: C++17 allows this as an extension for compatibility
  explicit(true) S(int);
  // expected-error@-1 {{explicit(bool) requires C++20 or later}}
  
  explicit(false) S(float);
  // expected-error@-1 {{explicit(bool) requires C++20 or later}}
};