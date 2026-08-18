// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Regression test for a compiler crash: a null QualType reaching the
// "Cannot retrieve a NULL type pointer" assert in getCommonPtr, via
// ConstructDynamicBoundType::VisitParenType.
//
// FIXME: The "pointer to incomplete __counted_by array type" message fires for
// a complete array here; that misleading wording is tracked by
// rdar://184258376.

#include <ptrcheck.h>

// __counted_by
// expected-error@+2{{pointer to incomplete __counted_by array type 'int *__bidi_indexable[10]' not allowed; did you mean to use a nested pointer type?}}
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void paren_counted_bidi(int * __bidi_indexable __counted_by(n) (*p)[10], int n);
// expected-error@+2{{pointer to incomplete __counted_by array type 'int *__indexable[10]' not allowed; did you mean to use a nested pointer type?}}
// expected-error@+1{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
void paren_counted_indexable(int * __indexable __counted_by(n) (*p)[10], int n);

// __sized_by
// expected-error@+2{{pointer to incomplete __counted_by array type 'char *__bidi_indexable[10]' not allowed; did you mean to use a nested pointer type?}}
// expected-error@+1{{pointer cannot be '__sized_by' and '__bidi_indexable' at the same time}}
void paren_sized_bidi(char * __bidi_indexable __sized_by(n) (*p)[10], int n);
// expected-error@+2{{pointer to incomplete __counted_by array type 'char *__indexable[10]' not allowed; did you mean to use a nested pointer type?}}
// expected-error@+1{{pointer cannot be '__sized_by' and '__indexable' at the same time}}
void paren_sized_indexable(char * __indexable __sized_by(n) (*p)[10], int n);

// The '_or_null' variants are rejected earlier ("array objects cannot be null")
// and do not reach VisitParenType, but are included for coverage of the family.
// expected-error@+1{{array objects cannot be null; did you mean __counted_by instead?}}
void paren_counted_or_null_bidi(int * __bidi_indexable __counted_by_or_null(n) (*p)[10], int n);
// expected-error@+1{{array objects cannot be null; did you mean __counted_by instead?}}
void paren_counted_or_null_indexable(int * __indexable __counted_by_or_null(n) (*p)[10], int n);
// expected-error@+1{{array objects cannot be null; did you mean __sized_by instead?}}
void paren_sized_or_null_bidi(char * __bidi_indexable __sized_by_or_null(n) (*p)[10], int n);
// expected-error@+1{{array objects cannot be null; did you mean __sized_by instead?}}
void paren_sized_or_null_indexable(char * __indexable __sized_by_or_null(n) (*p)[10], int n);

// The '__ended_by' variants hit a pre-existing assert and are in
// `clang/test/BoundsSafety/Sema/crash-ended-by-paren-type.c`.
