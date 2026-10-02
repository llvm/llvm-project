// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {};

void test(int a, struct Foo b, int *d, struct Foo *e, const struct Foo *f) {
  __builtin_clear_padding(a); // expected-error {{passing 'int' to parameter of incompatible type pointer: type mismatch at 1st parameter ('int' vs pointer)}}
  __builtin_clear_padding(b); // expected-error {{passing 'struct Foo' to parameter of incompatible type pointer: type mismatch at 1st parameter ('struct Foo' vs pointer)}}
  __builtin_clear_padding(d); // This should not error.
  __builtin_clear_padding(e); // This should not error.
  __builtin_clear_padding(f); // expected-error {{read-only variable is not assignable}}
}

struct Incomplete; // expected-note {{forward declaration of 'struct Incomplete'}}

void testIncomplete(void* v, struct Incomplete *i) {
  __builtin_clear_padding(v); // expected-error {{variable has incomplete type 'void'}}
  __builtin_clear_padding(i); // expected-error {{variable has incomplete type 'struct Incomplete'}}
}

void testNumArgs(int* i) {
  __builtin_clear_padding(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_clear_padding(i); // This should not error.
  __builtin_clear_padding(i, i); // expected-error {{too many arguments to function call, expected 1, have 2}}
  __builtin_clear_padding(i, i, i); // expected-error {{too many arguments to function call, expected 1, have 3}}
  __builtin_clear_padding(i, i, i, i); // expected-error {{too many arguments to function call, expected 1, have 4}}
}

void testFunctionPointer(void(*f)()) {
  __builtin_clear_padding(f); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('void (*)()' invalid)}}
}

struct WithVLA {
  int i;
  char c[];
};

struct WithVLA2 {
  int i2;
  struct WithVLA w;
};

struct WithVLA3 {
  struct WithVLA2 w2;
};

void testVLA(struct WithVLA* w1, struct WithVLA2* w2, struct WithVLA3* w3) {
  __builtin_clear_padding(w1); // expected-error {{'struct WithVLA' has flexible array member, which is unsupported by __builtin_clear_padding}}
  __builtin_clear_padding(w2); // expected-error {{'struct WithVLA2' has flexible array member, which is unsupported by __builtin_clear_padding}}
  __builtin_clear_padding(w3); // expected-error {{'struct WithVLA3' has flexible array member, which is unsupported by __builtin_clear_padding}}
}

