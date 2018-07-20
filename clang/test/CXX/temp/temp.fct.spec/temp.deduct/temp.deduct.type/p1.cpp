// RUN: %clang_cc1 -verify %s

// an attempt is made to find template argument values that will make P, after
// substitution of the deduced values, compatible with A

namespace cv_mismatch {
  template<typename> struct X {};
  template<typename T> void f(X<const T>); // expected-note {{cannot deduce a type for 'T' that would make 'const T' equal 'volatile int'}}
  void g() { f(X<volatile int>()); } // expected-error {{no matching}}
}
