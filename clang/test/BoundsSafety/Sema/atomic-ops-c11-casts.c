
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void test(void) {
  // expected-error@+4{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr1 <- val1
  int *_Atomic __single ptr1;
  int *__unsafe_indexable val1;
  __c11_atomic_store(&ptr1, val1, __ATOMIC_SEQ_CST);

  // expected-error@+4{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr2 <- val2
  int *_Atomic __single ptr2;
  int *__unsafe_indexable val2;
  __c11_atomic_exchange(&ptr2, val2, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*'; use explicit cast to perform this conversion}}
  // expected3 <- ptr3
  int *_Atomic __unsafe_indexable ptr3;
  int *__single expected3;
  int *__unsafe_indexable desired3;
  __c11_atomic_compare_exchange_strong(&ptr3, &expected3, desired3, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr4 <- desired4
  int *_Atomic __single ptr4;
  int *__single expected4;
  int *__unsafe_indexable desired4;
  __c11_atomic_compare_exchange_strong(&ptr4, &expected4, desired4, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*'; use explicit cast to perform this conversion}}
  // expected5 <- ptr5
  int *_Atomic __unsafe_indexable ptr5;
  int *__single expected5;
  int *__unsafe_indexable desired5;
  __c11_atomic_compare_exchange_weak(&ptr5, &expected5, desired5, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr6 <- desired6
  int *_Atomic __single ptr6;
  int *__single expected6;
  int *__unsafe_indexable desired6;
  __c11_atomic_compare_exchange_weak(&ptr6, &expected6, desired6, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
