// RUN: %clang_cc1 -finclude-default-header -fsyntax-only -verify -triple dxil-unknown-shadermodel6.6-library %s

typedef float int8_t4_packed; // expected-error {{cannot combine with previous 'float' declaration specifier}} expected-warning {{typedef requires a name}}
typedef float uint8_t4_packed; // expected-error {{cannot combine with previous 'float' declaration specifier}} expected-warning {{typedef requires a name}}

void f(int8_t4_packed s_arg, uint8_t4_packed u_arg) {
  // Ensure we are only allowing packed type <-> uint
  uint a = (float)s_arg; // expected-error {{C-style cast from 'int8_t4_packed' to 'float' is not allowed}}
  uint b = (float)u_arg; // expected-error {{C-style cast from 'uint8_t4_packed' to 'float' is not allowed}}
  int c = s_arg; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'int8_t4_packed'}}
  int d = u_arg; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'uint8_t4_packed'}}
  float f1 = s_arg; // expected-error {{cannot initialize a variable of type 'float' with an lvalue of type 'int8_t4_packed'}}
  float f2 = u_arg; // expected-error {{cannot initialize a variable of type 'float' with an lvalue of type 'uint8_t4_packed'}}
}
