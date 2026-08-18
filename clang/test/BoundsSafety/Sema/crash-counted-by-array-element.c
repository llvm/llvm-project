// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=bounds-safety %s

// Regression test for a compiler crash (reachable assertion) in
// ConstructCountAttributedType::TraverseArrayType. When a '__counted_by' /
// '__sized_by' attribute targets the *element* of an array, type construction
// descends into the array element, rebuilds it, and used to abort with:
//   assert(T->getPointeeType() == NewElementTy &&
//          "pre-check should have rejected count on array element")
//
// For the array shape used here, -fexperimental-bounds-safety-attributes
// reaches the crashing TraverseArrayType path (checked by the 'expected'
// directives), while -fbounds-safety rejects the same source earlier, before
// type construction, with a different diagnostic (checked by the
// 'bounds-safety' directives). The two RUN lines pin both behaviours; this is
// why the crash reproducer uses -fexperimental-bounds-safety-attributes.


#include <ptrcheck.h>

// '__counted_by' on the element of the (decayed) array.
// FIXME: Saying "incomplete" here is misleading (rdar://184258376)
// expected-error@+4{{pointer to incomplete __counted_by array type 'int *[10]' not allowed; did you mean to use a nested pointer type?}}
// FIXME: This diagnostic is just confusing. We probably shouldn't emit it (rdar://184258982).
// expected-error@+2{{multiple coupled declarations in a -fbounds-safety attribute are not supported yet}}
// bounds-safety-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
void f_counted(int * __counted_by(n) p[5][10], int n);

// Same shape with '__sized_by'.
// expected-error@+3{{pointer to incomplete __counted_by array type 'char *[10]' not allowed; did you mean to use a nested pointer type?}}
// expected-error@+2{{multiple coupled declarations in a -fbounds-safety attribute are not supported yet}}
// bounds-safety-error@+1{{'__sized_by' attribute on nested pointer type is only allowed on indirect parameters}}
void f_sized(char * __sized_by(n) p[5][10], int n);

// The '_or_null' variants never reach TraverseArrayType and never crashed: under
// -fexperimental-bounds-safety-attributes they are rejected earlier
// (err_bounds_safety_nullable_fam, "array objects cannot be null"), and under
// -fbounds-safety they hit the same nested-pointer diagnostic as above. Included
// for exhaustiveness over the counted_by/sized_by family.
// expected-error@+2{{array objects cannot be null; did you mean __counted_by instead?}}
// bounds-safety-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void f_counted_or_null(int * __counted_by_or_null(n) p[5][10], int n);

// expected-error@+2{{array objects cannot be null; did you mean __sized_by instead?}}
// bounds-safety-error@+1{{'__sized_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void f_sized_or_null(char * __sized_by_or_null(n) p[5][10], int n);

// The '__ended_by' variant of this shape hits an assert and are in
// `crash-ended-by-array-element.c`
