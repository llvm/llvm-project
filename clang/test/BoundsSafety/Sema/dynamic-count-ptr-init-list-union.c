

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

union foo {
  struct {
    int *__counted_by(len) p;
    int len;
  } v1;
  struct {
    int dummy;
    int *__counted_by(len) p;
    int len;
  } v2;
};

void test_foo(void) {
  int arr[] = { 1, 2, 3 }; // expected-note 2{{'arr' declared here}}

  union foo f0;

  union foo f1 = {};

  union foo f2 = { .v1 = {} };

  union foo f3 = { .v2 = {} };

  union foo f4 = { .v1 = { .p = arr, .len = 3 } };
  union foo f5 = { .v1 = { .p = arr, .len = 4 } }; // expected-error{{initializing 'f5.v1.p' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with array 'arr' (which has 3 elements) always fails}}

  union foo f6 = { .v2 = { .dummy = 42, .p = arr, .len = 3 } };
  union foo f7 = { .v2 = { .dummy = 42, .p = arr, .len = 4 } }; // expected-error{{initializing 'f7.v2.p' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with array 'arr' (which has 3 elements) always fails}}
}

union bar {
  struct {
    int *__counted_by(len) p;
    int len;
  } v1;
  struct {
    int dummy;
    int *__counted_by(len + 1) p;
    int len;
  } v2;
};

void test_bar(void) {
  // expected-error@+1{{implicitly initializing 'b0.v2.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  union bar b0;

  // ok (clang picks the first field -- v1)
  // TODO: Should we emit an error anyway?
  union bar b1 = {};

  union bar b2 = { .v1 = {} };

  // expected-error@+1{{implicitly initializing 'b3.v2.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  union bar b3 = { .v2 = {} };
}

union baz {
  struct {
    int *__counted_by(len + 1) p;
    int len;
  } v1;
  struct {
    int dummy;
    int *__counted_by(len) p;
    int len;
  } v2;
};

void test_baz(void) {
  // expected-error@+1{{implicitly initializing 'b0.v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  union baz b0;

  // expected-error@+1{{implicitly initializing 'b1.v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  union baz b1 = {};

  // expected-error@+1{{implicitly initializing 'b2.v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  union baz b2 = { .v1 = {} };

  union baz b3 = { .v2 = {} };
}

struct qux {
  union {
    struct {
      int *__counted_by(len) p;
      int len;
    } v1;
    struct {
      int dummy;
      int *__counted_by(len + 1) p;
      int len;
    } v2;
  };
};

void test_qux(void) {
  // expected-error@+1{{implicitly initializing 'q0..v2.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct qux q0;

  // expected-error@+1{{implicitly initializing 'q1..v2.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct qux q1 = {};

  struct qux q2 = { .v1 = {} };

  // expected-error@+1{{implicitly initializing 'q3..v2.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct qux q3 = { .v2 = {} };
}

struct quux {
  union {
    struct {
      int *__counted_by(len + 1) p;
      int len;
    } v1;
    struct {
      int dummy;
      int *__counted_by(len) p;
      int len;
    } v2;
  };
};

void test_quux(void) {
  // expected-error@+1{{implicitly initializing 'q0..v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct quux q0;

  // expected-error@+1{{implicitly initializing 'q1..v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct quux q1 = {};

  // expected-error@+1{{implicitly initializing 'q2..v1.p' of type 'int *__single __counted_by(len + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
  struct quux q2 = { .v1 = {} };

  struct quux q3 = { .v2 = {} };
}
