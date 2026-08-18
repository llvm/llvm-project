// This test relies on the assertion firing, so it is only meaningful on an
// asserts build.
// REQUIRES: asserts
// XFAIL: *
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s

// Regression test for a compiler assert (rdar://184264642)

#include <ptrcheck.h>

// '__ended_by' on the element of the (decayed) array.
// FIXME: Saying "incomplete" here is misleading (rdar://184258376).
// expected-error@+1{{pointer to incomplete __counted_by array type 'int *[10]' not allowed; did you mean to use a nested pointer type?}}
void f_ended(int * __ended_by(e) p[5][10], int *e);
