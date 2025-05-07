
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include "atomic-ops-gnu-system.h"

void unsafe_indexable(void) {
  int x;
  int *__unsafe_indexable p = &x;
  int *__unsafe_indexable q;
  int *__unsafe_indexable r;

  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__unsafe_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void single(void) {
  int x;
  int *__single p = &x;
  int *__single q;
  int *__single r;

  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single*__bidi_indexable' invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single*__bidi_indexable' invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single*__bidi_indexable' invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void indexable(void) {
  int x;
  int *__indexable p = &x;
  int *__indexable q;
  int *__indexable r;

  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__indexable' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void bidi_indexable(void) {
  int x;
  int *__bidi_indexable p = &x;
  int *__bidi_indexable q;
  int *__bidi_indexable r;

  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__bidi_indexable' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__bidi_indexable*__bidi_indexable' invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void counted_by(void) {
  int x;
  int *__counted_by(1) p = &x;
  int *__counted_by(1) q = &x;
  int *__counted_by(1) r = &x;

  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void sized_by(void) {
  int x;
  int *__sized_by(4) p = &x;
  int *__sized_by(4) q = &x;
  int *__sized_by(4) r = &x;

  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void counted_by_or_null(void) {
  int x;
  int *__counted_by_or_null(1) p = &x;
  int *__counted_by_or_null(1) q = &x;
  int *__counted_by_or_null(1) r = &x;

  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__counted_by_or_null' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __counted_by_or_null(1)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void sized_by_or_null(void) {
  int x;
  int *__sized_by_or_null(4) p = &x;
  int *__sized_by_or_null(4) q = &x;
  int *__sized_by_or_null(4) r = &x;

  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__sized_by_or_null' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __sized_by_or_null(4)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void ended_by(int * x, int * y, int * z, int * __ended_by(x) p, int * __ended_by(y) q, int *__ended_by(z) r) {
  // expected-error@+2{{assignment to 'int *__single __ended_by(y)' (aka 'int *__single') 'q' requires corresponding assignment to 'y'; add self assignment 'y = y' if the value has not changed}}
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  q = __atomic_load_n(&p, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_load(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_store_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_store(&p, &q, __ATOMIC_SEQ_CST);

  // expected-error@+2{{assignment to 'int *__single __ended_by(y)' (aka 'int *__single') 'q' requires corresponding assignment to 'y'; add self assignment 'y = y' if the value has not changed}}
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  q = __atomic_exchange_n(&p, q, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_exchange(&p, &q, &r, __ATOMIC_SEQ_CST);

  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_compare_exchange_n(&p, &q, r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  // expected-error@+1{{atomic operation on '__ended_by' pointer is not yet supported}}
  __atomic_compare_exchange(&p, &q, &r, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_add_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_sub_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int *__single __ended_by(x)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}
