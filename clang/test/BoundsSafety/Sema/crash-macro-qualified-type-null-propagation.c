// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Regression test for a compiler crash: a null QualType reaching the
// "Cannot retrieve a NULL type pointer" assert in getCommonPtr, via
// ConstructDynamicBoundType::VisitMacroQualifiedType.

#include <ptrcheck.h>

#define NODEREF __attribute__((noderef))

// __counted_by
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void macro_counted_bidi(int * __bidi_indexable __counted_by(n) NODEREF * p, int n);
// expected-error@+1{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
void macro_counted_indexable(int * __indexable __counted_by(n) NODEREF * p, int n);

// __counted_by_or_null
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__bidi_indexable' at the same time}}
void macro_counted_or_null_bidi(int * __bidi_indexable __counted_by_or_null(n) NODEREF * p, int n);
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__indexable' at the same time}}
void macro_counted_or_null_indexable(int * __indexable __counted_by_or_null(n) NODEREF * p, int n);

// __sized_by
// expected-error@+1{{pointer cannot be '__sized_by' and '__bidi_indexable' at the same time}}
void macro_sized_bidi(char * __bidi_indexable __sized_by(n) NODEREF * p, int n);
// expected-error@+1{{pointer cannot be '__sized_by' and '__indexable' at the same time}}
void macro_sized_indexable(char * __indexable __sized_by(n) NODEREF * p, int n);

// __sized_by_or_null
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__bidi_indexable' at the same time}}
void macro_sized_or_null_bidi(char * __bidi_indexable __sized_by_or_null(n) NODEREF * p, int n);
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__indexable' at the same time}}
void macro_sized_or_null_indexable(char * __indexable __sized_by_or_null(n) NODEREF * p, int n);

// __ended_by (here the ended_by pointer is at level 1, so this does not hit the
// pre-existing 'assert(Level <= 1)' crash; see rdar://184264642)
// expected-error@+1{{pointer cannot be '__ended_by' and '__bidi_indexable' at the same time}}
void macro_ended_bidi(int * __bidi_indexable __ended_by(e) NODEREF * p, int *e);
// expected-error@+1{{pointer cannot be '__ended_by' and '__indexable' at the same time}}
void macro_ended_indexable(int * __indexable __ended_by(e) NODEREF * p, int *e);
