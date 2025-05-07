
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include "atomic-ops-c11-system.h"

void unsafe_indexable(void) {
  int x;
  int *_Atomic __unsafe_indexable p = &x;
  int *__unsafe_indexable q;

  p++;
  p--;
  ++p;
  --p;

  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *__unsafe_indexable)' and 'int'}}
  p += 42;
  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *__unsafe_indexable)' and 'int'}}
  p -= 42;

  q = &p[42];

  __c11_atomic_init(&p, &x);
  q = __c11_atomic_load(&p, __ATOMIC_SEQ_CST);
  __c11_atomic_store(&p, q, __ATOMIC_SEQ_CST);

  q = __c11_atomic_exchange(&p, q, __ATOMIC_SEQ_CST);
  __c11_atomic_compare_exchange_strong(&p, &q, q, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  __c11_atomic_compare_exchange_weak(&p, &q, q, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  q = __c11_atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);
  q = __c11_atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__unsafe_indexable) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__unsafe_indexable) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__unsafe_indexable) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *__unsafe_indexable) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *__unsafe_indexable) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}

void single(void) {
  int x;
  int *_Atomic __single p = &x;
  int *__single q;

  // expected-error-re@+1{{pointer arithmetic on single pointer 'p' is out of {{bounds$}}}}
  p++;
  // expected-error-re@+1{{pointer arithmetic on single pointer 'p' is out of {{bounds$}}}}
  p--;
  // expected-error-re@+1{{pointer arithmetic on single pointer 'p' is out of {{bounds$}}}}
  ++p;
  // expected-error-re@+1{{pointer arithmetic on single pointer 'p' is out of {{bounds$}}}}
  --p;

  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *__single)' and 'int'}}
  p += 42;
  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *__single)' and 'int'}}
  p -= 42;

  // expected-error@+1{{array subscript on single pointer 'p' must use a constant index of 0 to be in bounds}}
  q = &p[42];

  __c11_atomic_init(&p, &x);
  q = __c11_atomic_load(&p, __ATOMIC_SEQ_CST);
  __c11_atomic_store(&p, q, __ATOMIC_SEQ_CST);

  q = __c11_atomic_exchange(&p, q, __ATOMIC_SEQ_CST);
  __c11_atomic_compare_exchange_strong(&p, &q, q, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  __c11_atomic_compare_exchange_weak(&p, &q, q, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('_Atomic(int *__single) *__bidi_indexable' invalid)}}
  q = __c11_atomic_fetch_add(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic arithmetic operation must be a pointer to '__unsafe_indexable' pointer ('_Atomic(int *__single) *__bidi_indexable' invalid)}}
  q = __c11_atomic_fetch_sub(&p, 42, __ATOMIC_SEQ_CST);

  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__single) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__single) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *__single) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *__single) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *__single) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}
