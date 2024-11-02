// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -finclude-default-header -verify %s

struct Undefined; // expected-note {{forward declaration of 'Undefined'}}
_Static_assert(!__builtin_hlsl_is_intangible(Undefined), ""); // expected-error{{incomplete type 'Undefined' used in type trait expression}}

void fn(int X) { // expected-note {{declared here}}
  // expected-error@#vla {{variable length arrays are not supported for the current target}}
  // expected-error@#vla {{variable length arrays are not supported in '__builtin_hlsl_is_intangible'}}
  // expected-warning@#vla {{variable length arrays in C++ are a Clang extension}}
  // expected-note@#vla {{function parameter 'X' with unknown value cannot be used in a constant expression}}
  _Static_assert(!__builtin_hlsl_is_intangible(int[X]), ""); // #vla
}
