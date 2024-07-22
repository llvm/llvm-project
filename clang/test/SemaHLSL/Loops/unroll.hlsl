// RUN: %clang_cc1 -O0 -finclude-default-header -fsyntax-only -triple dxil-pc-shadermodel6.6-library %s -verify
void unroll_no_vars() {
  // expected-note@+1 {{declared here}}
  int I = 3;
  // expected-error@+2 {{expression is not an integral constant expression}}
  // expected-note@+1 {{read of non-const variable 'I' is not allowed in a constant expression}}
  [unroll(I)]
  while (I--);
}

void unroll_arg_count() {
   [unroll(2,4)] // expected-error {{'unroll' attribute takes no more than 1 argument}}
  for(int i=0; i<100; i++);
}

void loop_arg_count() {
   [loop(2)] // expected-error {{'loop' attribute takes no more than 0 argument}}
  for(int i=0; i<100; i++);
}

void unroll_no_negative() {
  [unroll(-1)] // expected-error {{invalid value '-1'; must be positive}}
  for(int i=0; i<100; i++);
}

void unroll_no_zero() {
  [unroll(0)] // expected-error {{invalid value '0'; must be positive}}
  for(int i=0; i<100; i++);
}

void unroll_no_float() {
  [unroll(2.1)] // expected-error {{invalid argument of type 'float'; expected an integer type}}
  for(int i=0; i<100; i++);
}

void unroll_no_bool_false() {
  [unroll(false)] // expected-error {{invalid argument of type 'bool'; expected an integer type}}
  for(int i=0; i<100; i++);
}

void unroll_no_bool_true() {
  [unroll(true)] // expected-error {{invalid argument of type 'bool'; expected an integer type}}
  for(int i=0; i<100; i++);
}

void unroll_loop_enforcement() {
  int x[10];
  [unroll(4)] // expected-error {{'unroll' attribute only applies to 'for', 'while', and 'do' statements}}
  if (x[0])
    x[0] = 15;
}
