// RUN: %clang_cc1 -verify -std=c++98 %s
// expected-no-diagnostics

// Objective-C allows C++11 enumerations in C++98 mode. We disambiguate in
// order to make this a backwards-compatible extension.
struct A {
  enum E : int{a}; // OK, enum definition
  enum E : int(a); // OK, bit-field declaration
};
_Static_assert(A::a == 0, "");
