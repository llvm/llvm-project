// RUN: %clang_cc1 -verify -fsyntax-only %s

// Tests for issue #207785.
//
// Check that a cleanup attribute on an invalid declaration doesn't crash,
// and that we diagnose duplicate cleanup attributes.

void f1(unsigned *x) {}
void b1() {
  __attribute__((cleanup(f1))) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f1))) // expected-error {{'cleanup' function 'f1' parameter has type 'unsigned int *' which is incompatible with type 'int *'}}
  baz8; // expected-error {{type specifier missing, defaults to 'int'}}

}

void b2() {
  __attribute__((cleanup(f1))) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f1))) // expected-error {{'cleanup' function 'f1' parameter has type 'unsigned int *' which is incompatible with type 'int *'}}
  int baz8;
}

void f2(double *x);
void f3(double *x);
void f4(double *x);
void b3() {
  __attribute__((cleanup(f2))) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f3))) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f4)))
  double x;
}

void b4() {
  [[gnu::cleanup(f2)]] // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  [[gnu::cleanup(f3)]]
  double x;
}

void b5() {
  [[gnu::cleanup(f2)]] // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f3))) // expected-warning {{declaration has multiple 'cleanup' attributes; all but the last one will be ignored}}
  __attribute__((cleanup(f4)))
  double x;
}
