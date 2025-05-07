
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

static int array[42];
static int nt_array[__null_terminated 42];

void ptr_type(int *__null_terminated tb, int *__single s, int *__bidi_indexable bi) {
  __terminated_by_to_indexable(array);    // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__bidi_indexable' invalid)}}
  __terminated_by_to_indexable(nt_array); // ok (the array decays to a __single __null_terminated pointer)

  __terminated_by_to_indexable(tb); // ok
  __terminated_by_to_indexable(s);  // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__single' invalid)}}
  __terminated_by_to_indexable(bi); // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__bidi_indexable' invalid)}}
  __terminated_by_to_indexable(*s); // expected-error{{pointer argument must be a '__terminated_by' pointer ('int' invalid)}}
  __terminated_by_to_indexable(0);  // expected-error{{pointer argument must be a '__terminated_by' pointer ('int' invalid)}}
}

void unsafe_ptr_type(int *__null_terminated tb, int *__single s, int *__bidi_indexable bi) {
  __unsafe_terminated_by_to_indexable(array);    // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__bidi_indexable' invalid)}}
  __unsafe_terminated_by_to_indexable(nt_array); // ok (the array decays to a __single __null_terminated pointer)

  __unsafe_terminated_by_to_indexable(tb); // ok
  __unsafe_terminated_by_to_indexable(s);  // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__single' invalid)}}
  __unsafe_terminated_by_to_indexable(bi); // expected-error{{pointer argument must be a '__terminated_by' pointer ('int *__bidi_indexable' invalid)}}
  __unsafe_terminated_by_to_indexable(*s); // expected-error{{pointer argument must be a '__terminated_by' pointer ('int' invalid)}}
  __unsafe_terminated_by_to_indexable(0);  // expected-error{{pointer argument must be a '__terminated_by' pointer ('int' invalid)}}
}

void term_ice(int *__null_terminated p, int val) {
  (void)__builtin_terminated_by_to_indexable(p, 0);      // ok
  (void)__builtin_terminated_by_to_indexable(p, 42 * 0); // ok
  (void)__builtin_terminated_by_to_indexable(p, val);    // expected-error{{terminator value is not a constant expression}}
}

void unsafe_term_ice(int *__null_terminated p, int val) {
  (void)__builtin_unsafe_terminated_by_to_indexable(p, 0);      // ok
  (void)__builtin_unsafe_terminated_by_to_indexable(p, 42 * 0); // ok
  (void)__builtin_unsafe_terminated_by_to_indexable(p, val);    // expected-error{{terminator value is not a constant expression}}
}

void null(int *__null_terminated p, int *__terminated_by(42) q) {
  __null_terminated_to_indexable(p); // ok
  __null_terminated_to_indexable(q); // expected-error{{pointer argument must be terminated by '0' (got '42')}}
}

void unsafe_null(int *__null_terminated p, int *__terminated_by(42) q) {
  __unsafe_null_terminated_to_indexable(p); // ok
  __unsafe_null_terminated_to_indexable(q); // expected-error{{pointer argument must be terminated by '0' (got '42')}}
}

void _42(int *__null_terminated p, int *__terminated_by(42) q) {
  (void)__builtin_terminated_by_to_indexable(p, 42); // expected-error{{pointer argument must be terminated by '42' (got '0')}}
  (void)__builtin_terminated_by_to_indexable(q, 42); // ok
}

void unsafe_42(int *__null_terminated p, int *__terminated_by(42) q) {
  (void)__builtin_unsafe_terminated_by_to_indexable(p, 42); // expected-error{{pointer argument must be terminated by '42' (got '0')}}
  (void)__builtin_unsafe_terminated_by_to_indexable(q, 42); // ok
}
