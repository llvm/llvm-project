// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {};

struct Incomplete; // expected-note {{forward declaration of 'struct Incomplete'}}

void test(int a, struct Foo b, void *c, int *d, struct Foo *e, const struct Foo *f, struct Incomplete *g) {
  __builtin_clear_padding(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_clear_padding(d, d); // expected-error {{too many arguments to function call, expected 1, have 2}}

  __builtin_clear_padding(a); // expected-error {{passing 'int' to parameter of incompatible type pointer: type mismatch at 1st parameter ('int' vs pointer)}}
  __builtin_clear_padding(b); // expected-error {{passing 'struct Foo' to parameter of incompatible type pointer: type mismatch at 1st parameter ('struct Foo' vs pointer)}}
  __builtin_clear_padding(c); // expected-error {{variable has incomplete type 'void'}}
  __builtin_clear_padding(d); // This should not error.
  __builtin_clear_padding(e); // This should not error.
  __builtin_clear_padding(f); // expected-error {{read-only variable is not assignable}}
  __builtin_clear_padding(g); // expected-error {{variable has incomplete type 'struct Incomplete'}}
}
