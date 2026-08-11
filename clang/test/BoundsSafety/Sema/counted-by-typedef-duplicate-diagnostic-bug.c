// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -verify %s

// FIXME: Port these checks into `Sema/nested-sugar-no-duplicate-diagnostics.c`.
// Originally this file documented a duplicate diagnostic bug that's now fixed.

#include <ptrcheck.h>

typedef int fn_t(int);
typedef fn_t *fnptr_t;

int len;

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'fn_t' (aka 'int (int)') is a function type}}
fn_t *bare __counted_by(len);

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'fn_t' (aka 'int (int)') is a function type}}
fnptr_t alias __counted_by(len);
