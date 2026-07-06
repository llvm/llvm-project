// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux -verify -fsyntax-only %s

// Check that a cleanup attribute on an invalid declaration doesn't crash,
// and that we diagnose duplicate cleanup attributes.

#define C(x) __attribute__((cleanup(x)))
void foo(double *x U) {} // expected-error {{expected ')'}} expected-note {{to match this '('}}
void bar() {
  C(foo) C(foo) baz8; // expected-error {{type specifier missing, defaults to 'int'}} \
                         expected-warning 2 {{passing 4-byte aligned argument to 8-byte aligned parameter}} \
                         expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}} \
                         expected-error {{'cleanup' function 'foo' parameter has type 'double *' which is incompatible with type 'int *'}}
}

void f1(double *x);
void f2(double *x);
void f3(double *x);
void bar2() {
  C(f1) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  C(f2) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  C(f3)
  double x;
}
