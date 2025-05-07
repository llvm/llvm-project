
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void test(void) {
  // expected-error@+4{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*'; use explicit cast to perform this conversion}}
  // ret1 <- ptr1
  int *__unsafe_indexable ptr1;
  int *__single ret1;
  __atomic_load(&ptr1, &ret1, __ATOMIC_SEQ_CST);

  // expected-error@+4{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr2 <- val2
  int *__single ptr2;
  int *__unsafe_indexable val2;
  __atomic_store_n(&ptr2, val2, __ATOMIC_SEQ_CST);

  // expected-error@+4{{passing 'int *__unsafe_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single*__bidi_indexable'; use explicit cast to perform this conversion}}
  // ptr3 <- val3
  int *__single ptr3;
  int *__unsafe_indexable val3;
  __atomic_store(&ptr3, &val3, __ATOMIC_SEQ_CST);

  // expected-error@+4{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr4 <- val4
  int *__single ptr4;
  int *__unsafe_indexable val4;
  __atomic_exchange_n(&ptr4, val4, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__unsafe_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single*'; use explicit cast to perform this conversion}}
  // ptr5 <- val5
  int *__single ptr5;
  int *__unsafe_indexable val5;
  int *__single ret5;
  __atomic_exchange(&ptr5, &val5, &ret5, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*__bidi_indexable'; use explicit cast to perform this conversion}}
  // ret6 <- ptr6
  int *__unsafe_indexable ptr6;
  int *__unsafe_indexable val6;
  int *__single ret6;
  __atomic_exchange(&ptr6, &val6, &ret6, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*'; use explicit cast to perform this conversion}}
  // expected7 <- ptr7
  int *__unsafe_indexable ptr7;
  int *__single expected7;
  int *__unsafe_indexable desired7;
  __atomic_compare_exchange_n(&ptr7, &expected7, desired7, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  // ptr8 <- desired8
  int *__single ptr8;
  int *__single expected8;
  int *__unsafe_indexable desired8;
  __atomic_compare_exchange_n(&ptr8, &expected8, desired8, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__single*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__unsafe_indexable*'; use explicit cast to perform this conversion}}
  // expected9 <- ptr9
  int *__unsafe_indexable ptr9;
  int *__single expected9;
  int *__unsafe_indexable desired9;
  __atomic_compare_exchange(&ptr9, &expected9, &desired9, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+5{{passing 'int *__unsafe_indexable*__bidi_indexable' to parameter of incompatible nested pointer type 'int *__single*__bidi_indexable'; use explicit cast to perform this conversion}}
  // ptr10 <- desired10
  int *__single ptr10;
  int *__single expected10;
  int *__unsafe_indexable desired10;
  __atomic_compare_exchange(&ptr10, &expected10, &desired10, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
