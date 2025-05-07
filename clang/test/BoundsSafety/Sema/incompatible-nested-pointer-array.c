
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

typedef struct lol {
  int *x[3];
} lol_t;

lol_t *g;

// expected-note@+1{{passing argument to parameter 'arg' here}}
void test_arg(int *__unsafe_indexable *__counted_by(3) arg);

void test(void) {
  // expected-error@+1{{initializing 'int *__unsafe_indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__single[3]'}}
  int *__unsafe_indexable *__counted_by(3) m1 = g->x;
  int *__unsafe_indexable *__counted_by(3) m2 = (int *__unsafe_indexable *)g->x;

  // expected-error@+1{{passing 'int *__single[3]' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*__single __counted_by(3)' (aka 'int *__unsafe_indexable*__single')}}
  test_arg(g->x);

  int *__unsafe_indexable *pp1;
  // expected-error@+1{{assigning to 'int *__unsafe_indexable*__bidi_indexable' from 'int *__single[3]'}}
  pp1 = g->x;
}

int *__unsafe_indexable * test_ret() {
  // expected-error@+1{{returning 'int *__single[3]' from a function with result of incompatible nested pointer type 'int *__unsafe_indexable*__single'}}
  return g->x;
}
