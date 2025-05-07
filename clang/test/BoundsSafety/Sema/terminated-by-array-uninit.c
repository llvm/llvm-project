
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Foo {
  int a[__null_terminated 2];
  int b[__terminated_by(42) 2];
};

struct Bar {
  int a[__null_terminated 2];
};

struct Baz {
  struct Foo f;
  struct Bar b;
};

// ok
int g_a[__null_terminated 8];

// expected-error@+1{{array 'g_b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
int g_b[__terminated_by(42) 8];

// expected-error@+1{{array 'g_foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
struct Foo g_foo;

// ok
struct Bar g_bar;

// expected-error@+1{{array 'g_baz.f.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
struct Baz g_baz;

void no_init_static_local(void) {
  // ok
  static int a[__null_terminated 8];

  // expected-error@+1{{array 'b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  static int b[__terminated_by(42) 8];

  // expected-error@+1{{array 'foo.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  static struct Foo foo;

  // ok
  static struct Bar bar;

  // expected-error@+1{{array 'baz.f.b' with '__terminated_by' attribute is initialized with an incorrect terminator (expected: '42'; got '0')}}
  static struct Baz baz;
}

void no_init_local(void) {
  // expected-error@+1{{array 'a' with '__terminated_by' attribute must be initialized}}
  int a[__null_terminated 8];

  // expected-error@+1{{array 'b' with '__terminated_by' attribute must be initialized}}
  int b[__terminated_by(42) 8];

  // expected-error@+2{{array 'foo.a' with '__terminated_by' attribute must be initialized}}
  // expected-error@+1{{array 'foo.b' with '__terminated_by' attribute must be initialized}}
  struct Foo foo;

  // expected-error@+1{{array 'bar.a' with '__terminated_by' attribute must be initialized}}
  struct Bar bar;

  // expected-error@+3{{array 'baz.f.a' with '__terminated_by' attribute must be initialized}}
  // expected-error@+2{{array 'baz.f.b' with '__terminated_by' attribute must be initialized}}
  // expected-error@+1{{array 'baz.b.a' with '__terminated_by' attribute must be initialized}}
  struct Baz baz;
}
