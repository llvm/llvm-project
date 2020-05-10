// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify=cxx11 -std=c++11 %s

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

// Objective-C allows C++11 enumerations in C++98 mode. We disambiguate in
// order to make this a backwards-compatible extension.
struct A {
  enum E : int{a}; // OK, enum definition
  enum E : int(a); // OK, bit-field declaration cxx11-error{{anonymous bit-field}}
};
_Static_assert(A::a == 0, "");
