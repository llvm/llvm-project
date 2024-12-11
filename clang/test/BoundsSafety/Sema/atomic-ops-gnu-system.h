
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

#pragma clang system_header

void unspecified(void) {
  int x;
  int * p = &x;
  int * q;
  int * r;

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

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_and_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_or_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_xor_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_nand_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_nand(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int **__bidi_indexable' invalid)}}
  __atomic_min_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int **__bidi_indexable' invalid)}}
  __atomic_max_fetch(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to integer or supported floating point type ('int **__bidi_indexable' invalid)}}
  __atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}
