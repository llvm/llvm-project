// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

void test(int a, Foo b, void *c, int *d, Foo *e, const Foo *f, Incomplete *g) {
  __builtin_clear_padding(a); // expected-error {{passing 'int' to parameter of incompatible type pointer: type mismatch at 1st parameter ('int' vs pointer)}}
  __builtin_clear_padding(b); // expected-error {{passing 'Foo' to parameter of incompatible type pointer: type mismatch at 1st parameter ('Foo' vs pointer)}}
  __builtin_clear_padding(c); // expected-error {{variable has incomplete type 'void'}}
  __builtin_clear_padding(d); // This should not error.
  __builtin_clear_padding(e); // This should not error.
  __builtin_clear_padding(f); // expected-error {{read-only variable is not assignable}}
  __builtin_clear_padding(g); // expected-error {{variable has incomplete type 'Incomplete'}}
}
