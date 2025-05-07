

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// expected-no-diagnostics

#include <ptrcheck.h>

typedef struct foo {
  int field;
} *__single foo_t;

extern void takes_foo(foo_t foo);

struct foo array_of_one[1];

void
call_takes_foo(void)
{
  __auto_type foo = array_of_one;
  takes_foo(foo);
}
