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

// The same conflict, but with the wide pointer hidden behind sugar (a typedef
// or __typeof__).

typedef int * __bidi_indexable bidi_ptr_t;
typedef int * __indexable indexable_ptr_t;
typedef char * __bidi_indexable char_bidi_ptr_t;
typedef char * __indexable char_indexable_ptr_t;

// __counted_by via typedef
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void td_counted_bidi(bidi_ptr_t __counted_by(n) p, int n);
// expected-error@+1{{pointer cannot be '__counted_by' and '__indexable' at the same time}}
void td_counted_indexable(indexable_ptr_t __counted_by(n) p, int n);

// __counted_by_or_null via typedef
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__bidi_indexable' at the same time}}
void td_counted_or_null_bidi(bidi_ptr_t __counted_by_or_null(n) p, int n);
// expected-error@+1{{pointer cannot be '__counted_by_or_null' and '__indexable' at the same time}}
void td_counted_or_null_indexable(indexable_ptr_t __counted_by_or_null(n) p, int n);

// __sized_by via typedef
// expected-error@+1{{pointer cannot be '__sized_by' and '__bidi_indexable' at the same time}}
void td_sized_bidi(char_bidi_ptr_t __sized_by(n) p, int n);
// expected-error@+1{{pointer cannot be '__sized_by' and '__indexable' at the same time}}
void td_sized_indexable(char_indexable_ptr_t __sized_by(n) p, int n);

// __sized_by_or_null via typedef
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__bidi_indexable' at the same time}}
void td_sized_or_null_bidi(char_bidi_ptr_t __sized_by_or_null(n) p, int n);
// expected-error@+1{{pointer cannot be '__sized_by_or_null' and '__indexable' at the same time}}
void td_sized_or_null_indexable(char_indexable_ptr_t __sized_by_or_null(n) p, int n);

// __ended_by via typedef
// expected-error@+1{{pointer cannot be '__ended_by' and '__bidi_indexable' at the same time}}
void td_ended_bidi(bidi_ptr_t __ended_by(e) p, int *e);
// expected-error@+1{{pointer cannot be '__ended_by' and '__indexable' at the same time}}
void td_ended_indexable(indexable_ptr_t __ended_by(e) p, int *e);

// __typeof__ of a wide-pointer global
int * __bidi_indexable g_bidi;
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void typeof_counted_bidi(__typeof__(g_bidi) __counted_by(n) p, int n);

// Double typedef
typedef bidi_ptr_t bidi_ptr2_t;
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void td2_counted_bidi(bidi_ptr2_t __counted_by(n) p, int n);

// const-qualified typedef
// expected-error@+1{{pointer cannot be '__counted_by' and '__bidi_indexable' at the same time}}
void const_td_counted_bidi(const bidi_ptr_t __counted_by(n) p, int n);
