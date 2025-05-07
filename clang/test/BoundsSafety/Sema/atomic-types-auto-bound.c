
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void test(void) {
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  int *_Atomic p1;
  // expected-error@+1{{_Atomic on '__bidi_indexable' pointer is not yet supported}}
  _Atomic(int *) p2;

  // The nested pointers should be __single.
  int *_Atomic *_Atomic __unsafe_indexable p3;
  _Atomic(int *) *_Atomic __unsafe_indexable p4;

  // There shouldn't be an error about __bidi_indexable, since the attribute
  // is replaced by __single when __counted_by is handled.
  int len;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *_Atomic __counted_by(len) p5;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  int *__counted_by(len) _Atomic p6;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *__counted_by(len)) p7;
  // expected-error@+1{{_Atomic on '__counted_by' pointer is not yet supported}}
  _Atomic(int *) __counted_by(len) p8;
}
