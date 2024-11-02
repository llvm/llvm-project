// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library  -finclude-default-header -verify %s

struct Undefined; // expected-note {{forward declaration of 'Undefined'}}
_Static_assert(!__builtin_hlsl_is_intangible(Undefined), ""); // expected-error{{incomplete type 'Undefined' used in type trait expression}}

void fn(int X) {
  // expected-error@#vla {{variable length arrays are not supported for the current target}}
  // expected-error@#vla {{variable length arrays are not supported in '__builtin_hlsl_is_intangible'}}
  // expected-warning@#vla {{variable length arrays in C++ are a Clang extension}}
  _Static_assert(!__builtin_hlsl_is_intangible(int[X]), ""); // #vla
}
