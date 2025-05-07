
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-note@+1{{'ints' declared here}}
int ints[16];
// expected-note@+1{{'chars' declared here}}
char chars[16];

struct CountInt {
  int *__counted_by(len) p;
  unsigned len;
};

// ok
struct CountInt ci1 = {.p = ints, .len = sizeof(ints) / sizeof(int)};

// ok
struct CountInt ci2 = {.p = ints, .len = sizeof(ints) / sizeof(int) - 1};

// expected-error@+1{{initializing 'ci3.p' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 17 with array 'ints' (which has 16 elements) always fails}}
struct CountInt ci3 = {.p = ints, .len = sizeof(ints) / sizeof(int) + 1};

// expected-error@+2{{initializing 'ci4.p' of type 'int *__single __counted_by(len)' (aka 'int *__single') and size value of 64 with array 'chars' (which has 16 bytes) always fails}}
// expected-warning@+1{{incompatible pointer types initializing 'int *__single __counted_by(len)' (aka 'int *__single') with an expression of type 'char[16]'}}
struct CountInt ci4 = {.p = chars, .len = sizeof(chars)};

// expected-warning@+1{{incompatible pointer types initializing 'int *__single __counted_by(len)' (aka 'int *__single') with an expression of type 'char[16]'}}
struct CountInt ci5 = {.p = chars, .len = sizeof(chars) / sizeof(int)};

struct CountChar {
  char *__counted_by(len) p;
  unsigned len;
};

// ok
struct CountChar cc1 = {.p = chars, .len = sizeof(chars)};

// expected-warning@+1{{incompatible pointer types initializing 'char *__single __counted_by(len)' (aka 'char *__single') with an expression of type 'int[16]'}}
struct CountChar cc2 = {.p = ints, .len = sizeof(ints) / sizeof(int)};

// expected-warning@+1{{incompatible pointer types initializing 'char *__single __counted_by(len)' (aka 'char *__single') with an expression of type 'int[16]'}}
struct CountChar cc3 = {.p = ints, .len = sizeof(ints)};

struct NestedCount {
  char buf[16];

  struct {
    char *__counted_by(len) p;
    unsigned len;
  } nested;
};

// ok
struct NestedCount ni1 = {.nested = {.p = ni1.buf, .len = sizeof(ni1.buf)}};
