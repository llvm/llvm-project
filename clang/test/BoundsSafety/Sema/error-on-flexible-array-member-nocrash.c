// RUN: %clang_cc1 -fbounds-safety -fsyntax-only -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -verify %s



#include <ptrcheck.h>

struct saddr {
  unsigned len;
  char fam[__counted_by(len)];
};

void test(struct saddr *sa, unsigned length) {
  // expected-error@+1{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
  sa = sa + 1;
  sa->len = length;
}
