// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

// The goal of this test is to check that duplicate diagnostics are not emitted.

// A wide pointer reached through three typedef layers conflicts with every
// count/bound attribute.
typedef int * __bidi_indexable bidi_t;
typedef bidi_t bidi2_t;
typedef bidi2_t bidi3_t;

// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void counted(bidi3_t __counted_by(n) p, int n);
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__bidi_indexable' at the same time}}
void counted_or_null(bidi3_t __counted_by_or_null(n) p, int n);
// expected-error@+1{{pointer cannot be '__sized_by' and '__bidi_indexable' at the same time}}
void sized(bidi3_t __sized_by(n) p, int n);
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__bidi_indexable' at the same time}}
void sized_or_null(bidi3_t __sized_by_or_null(n) p, int n);
// expected-error@+1{{pointer cannot be '__ended_by' and '__bidi_indexable' at the same time}}
void ended(bidi3_t __ended_by(e) p, int *e);

// A pointer with an unknown-size pointee ('void') reached through three typedef
// layers. Unlike the conflict above, `ValidateBoundsAttrTypeShape` emits and
// then *continues* (recovering by treating the count as a byte count).
typedef void * vp_t;
typedef vp_t vp2_t;
typedef vp2_t vp3_t;

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
void counted_void(vp3_t __counted_by(n) p, int n);
// expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
void counted_or_null_void(vp3_t __counted_by_or_null(n) p, int n);
