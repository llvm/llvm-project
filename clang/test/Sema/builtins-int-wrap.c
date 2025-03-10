// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-unknown %s


struct Foo { int x; };

int test_add_nuw(int x, double y, struct Foo z) {
  __builtin_add_nuw(x, y); // expected-error {{both arguments must be of integral type but 2nd argument is 'double'}}
  __builtin_add_nuw(y, x); // expected-error {{both arguments must be of integral type but 1st argument is 'double'}}
  __builtin_add_nuw(x, z); // expected-error {{both arguments must be of integral type but 2nd argument is 'struct Foo'}}
}

int test_add_nsw(int x, double y, struct Foo z) {
  __builtin_add_nsw(x, y); // expected-error {{both arguments must be of integral type but 2nd argument is 'double'}}
  __builtin_add_nsw(y, x); // expected-error {{both arguments must be of integral type but 1st argument is 'double'}}
  __builtin_add_nsw(x, z); // expected-error {{both arguments must be of integral type but 2nd argument is 'struct Foo'}}
}

int test_add_nuw_nsw(int x, double y, struct Foo z) {
  __builtin_add_nuw_nsw(x, y); // expected-error {{both arguments must be of integral type but 2nd argument is 'double'}}
  __builtin_add_nuw_nsw(y, x); // expected-error {{both arguments must be of integral type but 1st argument is 'double'}}
  __builtin_add_nuw_nsw(x, z); // expected-error {{both arguments must be of integral type but 2nd argument is 'struct Foo'}}
}
