
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#pragma clang system_header

#include <ptrcheck.h>

void unspecified(void) {
  int x;
  int *_Atomic p = &x;
  int *q;

  p++;
  p--;
  ++p;
  --p;

  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *)' and 'int'}}
  p += 42;
  // expected-error@+1{{invalid operands to binary expression ('_Atomic(int *)' and 'int'}}
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

  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_and(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_or(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer ('_Atomic(int *) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_xor(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_min(&p, 42, __ATOMIC_SEQ_CST);
  // expected-error@+1{{address argument to atomic operation must be a pointer to atomic integer or supported floating point type ('_Atomic(int *) *__bidi_indexable' invalid}}
  __c11_atomic_fetch_max(&p, 42, __ATOMIC_SEQ_CST);
}
