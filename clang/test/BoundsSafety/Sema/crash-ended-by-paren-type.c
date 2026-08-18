// This test relies on the assertion firing, so it is only meaningful on an
// asserts build.
// REQUIRES: asserts
// XFAIL: *
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Regression test for a compiler assert (rdar://184264642)

#include <ptrcheck.h>

// '__ended_by' with an explicit pointer attribute inside a parenthesized
// declarator.
// FIXME: The "pointer to incomplete __counted_by array type" message fires for a
// complete array here; that misleading wording is tracked by rdar://184258376.
// expected-error@+1{{pointer to incomplete __counted_by array type 'int *__bidi_indexable[10]' not allowed; did you mean to use a nested pointer type?}}
void paren_ended_bidi(int * __bidi_indexable __ended_by(e) (*p)[10], int *e);
// expected-error@+1{{pointer to incomplete __counted_by array type 'int *__indexable[10]' not allowed; did you mean to use a nested pointer type?}}
void paren_ended_indexable(int * __indexable __ended_by(e) (*p)[10], int *e);
