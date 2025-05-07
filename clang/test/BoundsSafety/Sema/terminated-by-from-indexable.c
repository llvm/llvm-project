
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct Foo {
  int x;
  int y;
};

struct Bar;

static int array[42];

void term_ice(int *__indexable ptr, int val) {
  __unsafe_terminated_by_from_indexable(0, ptr);         // ok
  __unsafe_terminated_by_from_indexable(42 * 1337, ptr); // ok
  __unsafe_terminated_by_from_indexable(val, ptr);       // expected-error{{terminator value is not a constant expression}}
}

void ptr_type(int *__null_terminated tb, int *__single s, int *__indexable i,
              int *__bidi_indexable bi, int *__unsafe_indexable ui,
              int *__counted_by(len) cb, int len) {
  __unsafe_null_terminated_from_indexable(array); // ok
  __unsafe_null_terminated_from_indexable(tb);    // ok (tb is a __single pointer)
  __unsafe_null_terminated_from_indexable(s);     // ok
  __unsafe_null_terminated_from_indexable(i);     // ok
  __unsafe_null_terminated_from_indexable(bi);    // ok
  __unsafe_null_terminated_from_indexable(ui);    // expected-error{{pointer argument must be a safe pointer ('int *__unsafe_indexable' invalid)}}
  __unsafe_null_terminated_from_indexable(cb);    // ok
  __unsafe_null_terminated_from_indexable(*s);    // expected-error{{pointer argument must be a safe pointer ('int' invalid)}}
  __unsafe_null_terminated_from_indexable(0);     // expected-error{{pointer argument must be a safe pointer ('int' invalid)}}
}

void pointee_type(float *__indexable f, int *__indexable i, char *__indexable c,
                  struct Foo *__indexable foo, struct Bar *__indexable bar,
                  float **__indexable pf, int **__indexable pi,
                  char **__indexable pc, void **__indexable pv,
                  struct Foo **__indexable pfoo,
                  struct Bar **__indexable pbar,
                  int *__unsafe_indexable *__indexable pui,
                  int *__bidi_indexable *__indexable pbi) {
  __unsafe_null_terminated_from_indexable(f);    // expected-error{{pointee type of the pointer argument must be an integer or a non-wide pointer}}
  __unsafe_null_terminated_from_indexable(i);    // ok
  __unsafe_null_terminated_from_indexable(c);    // ok
  __unsafe_null_terminated_from_indexable(foo);  // expected-error{{pointee type of the pointer argument must be an integer or a non-wide pointer}}
  __unsafe_null_terminated_from_indexable(bar);  // expected-error{{pointee type of the pointer argument must be an integer or a non-wide pointer}}
  __unsafe_null_terminated_from_indexable(pf);   // ok
  __unsafe_null_terminated_from_indexable(pi);   // ok
  __unsafe_null_terminated_from_indexable(pc);   // ok
  __unsafe_null_terminated_from_indexable(pv);   // ok
  __unsafe_null_terminated_from_indexable(pfoo); // ok
  __unsafe_null_terminated_from_indexable(pbar); // ok
  __unsafe_null_terminated_from_indexable(pui);  // ok
  __unsafe_null_terminated_from_indexable(pbi);  // expected-error{{pointee type of the pointer argument must be an integer or a non-wide pointer}}
}

void ptr_to_term_type(int *__null_terminated tb, int *__single s,
                      int *__indexable i, int *__bidi_indexable bi,
                      int *__unsafe_indexable ui, int *__counted_by(len) cb,
                      int len) {
  __unsafe_null_terminated_from_indexable(i, array); // ok
  __unsafe_null_terminated_from_indexable(i, tb);    // ok
  __unsafe_null_terminated_from_indexable(i, s);     // ok
  __unsafe_null_terminated_from_indexable(i, i);     // ok
  __unsafe_null_terminated_from_indexable(i, bi);    // ok
  __unsafe_null_terminated_from_indexable(i, ui);    // ok
  __unsafe_null_terminated_from_indexable(i, cb);    // ok
  __unsafe_null_terminated_from_indexable(i, *s);    // expected-error{{pointer to terminator argument must be a pointer ('int' invalid)}}
  __unsafe_null_terminated_from_indexable(i, 0);     // expected-error{{pointer to terminator argument must be a pointer ('int' invalid)}}
}

void pointee_mismatch(float *__indexable f, int *__indexable i,
                      char *__indexable c, int **__indexable pi,
                      void **__indexable pv) {
  __unsafe_null_terminated_from_indexable(array, array); // ok
  __unsafe_null_terminated_from_indexable(i, f);         // expected-error{{pointee types of the pointer and pointer to terminator arguments must be the same}}
  __unsafe_null_terminated_from_indexable(i, i);         // ok
  __unsafe_null_terminated_from_indexable(i, c);         // expected-error{{pointee types of the pointer and pointer to terminator arguments must be the same}}
  __unsafe_null_terminated_from_indexable(i, pi);        // expected-error{{pointee types of the pointer and pointer to terminator arguments must be the same}}
  __unsafe_null_terminated_from_indexable(pi, pi);       // ok
  __unsafe_null_terminated_from_indexable(pv, pv);       // ok
  __unsafe_null_terminated_from_indexable(pi, pv);       // expected-error{{pointee types of the pointer and pointer to terminator arguments must be the same}}
}
