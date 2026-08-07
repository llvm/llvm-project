// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Regression test for a compiler crash (reachable assertion) in
// ConstructDynamicBoundType::VisitPointerType.

#include <ptrcheck.h>

// FIXME: This diagnostic is confusing (rdar://184349713).

// __counted_by
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void counted_bidi(int * __counted_by(n) * __bidi_indexable p, int n);
// expected-error@+1{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
void counted_indexable(int * __counted_by(n) * __indexable p, int n);

// __counted_by_or_null
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__bidi_indexable' at the same time}}
void counted_or_null_bidi(int * __counted_by_or_null(n) * __bidi_indexable p, int n);
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__indexable' at the same time}}
void counted_or_null_indexable(int * __counted_by_or_null(n) * __indexable p, int n);

// __sized_by
// expected-error@+1{{pointer cannot be '__sized_by' and '__bidi_indexable' at the same time}}
void sized_bidi(char * __sized_by(n) * __bidi_indexable p, int n);
// expected-error@+1{{pointer cannot be '__sized_by' and '__indexable' at the same time}}
void sized_indexable(char * __sized_by(n) * __indexable p, int n);

// __sized_by_or_null
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__bidi_indexable' at the same time}}
void sized_or_null_bidi(char * __sized_by_or_null(n) * __bidi_indexable p, int n);
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__indexable' at the same time}}
void sized_or_null_indexable(char * __sized_by_or_null(n) * __indexable p, int n);

// __ended_by
// expected-error@+1{{pointer cannot be '__ended_by' and '__bidi_indexable' at the same time}}
void ended_bidi(int * __ended_by(e) * __bidi_indexable p, int *e);
// expected-error@+1{{pointer cannot be '__ended_by' and '__indexable' at the same time}}
void ended_indexable(int * __ended_by(e) * __indexable p, int *e);
