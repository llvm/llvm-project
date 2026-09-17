// RUN: %clang_cc1 -finclude-default-header -fsyntax-only -verify -triple dxil-unknown-shadermodel6.6-library %s

typedef float int8_t4_packed; // expected-error {{cannot combine with previous 'float' declaration specifier}} expected-warning {{typedef requires a name}}
typedef float uint8_t4_packed; // expected-error {{cannot combine with previous 'float' declaration specifier}} expected-warning {{typedef requires a name}}

void f(int8_t4_packed s_arg, uint8_t4_packed u_arg) {
  int8_t4_packed s1;
  int8_t4_packed s2[10];
  uint8_t4_packed u1;
  uint8_t4_packed u2[10];

  uint32_t b = s_arg;
  uint32_t d = u_arg;
  int a = s_arg;
  int c = u_arg;
  int8_t4_packed u_to_s = u_arg;
  uint8_t4_packed s_to_u = s_arg;
  float f1 = s_arg;
  float f2 = u_arg;
}
