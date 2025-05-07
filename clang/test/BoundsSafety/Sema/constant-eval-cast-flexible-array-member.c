

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  int len;
  unsigned fam[__counted_by(len)];
};

struct S *g_s = (struct S *)(unsigned char[1]){}; // expected-error{{initializer element is not a compile-time constant}}

int f_s(struct S *);

void foo(void) {
  int result = f_s((struct S *)(unsigned char[1]){}); // ok
}
