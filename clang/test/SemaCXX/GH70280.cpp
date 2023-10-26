// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR70280 {
  typedef a; // expected-error {{a type specifier is required for all declarations}}
  using b = char*;
  template <typename... c> void d(c...) = d<b, a>(0, ""); // expected-error {{no matching function for call to 'd'}}
  // expected-error@-1 {{illegal initializer (only variables can be initialized)}}
  // expected-note@-2 {{candidate function template not viable: no known conversion from 'const char[1]' to 'a' (aka 'int') for 2nd argument}}
}