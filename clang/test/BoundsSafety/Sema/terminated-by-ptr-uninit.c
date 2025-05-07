
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-no-diagnostics

struct Foo {
  int *__null_terminated a;
  int *__terminated_by(42) b;
};

struct Bar {
  int *__null_terminated a;
};

struct Baz {
  struct Foo f;
  struct Bar b;
};

int *__null_terminated g_a; // ok
int *__null_terminated g_b; // ok
struct Foo g_foo;           // ok
struct Bar g_bar;           // ok
struct Baz g_baz;           // ok

void no_init_static_local(void) {
  static int *__null_terminated a; // ok
  static int *__null_terminated b; // ok
  static struct Foo foo;           // ok
  static struct Bar bar;           // ok
  static struct Baz baz;           // ok
}

void no_init_local(void) {
  int *__null_terminated a; // ok
  int *__null_terminated b; // ok
  struct Foo foo; // ok
  struct Bar bar; // ok
  struct Baz baz; // ok
}

void empty_init_local(void) {
  struct Foo foo = {}; // ok
  struct Bar bar = {}; // ok
  struct Baz baz = {}; // ok
}
