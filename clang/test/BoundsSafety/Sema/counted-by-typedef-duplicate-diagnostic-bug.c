// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -verify %s

// This test documents a bug introduced. This bug will be fixed when
// `Sema::ValidateBoundsAttrTypeShape` fully handles desugaring itself.

// A bounds attribute on a pointer whose pointee has unknown size (here a
// function type) is correctly rejected with a *single* diagnostic when written
// on a bare pointer. When the same pointer is reached through a typedef the
// diagnostic is incorrectly emitted twice.
//
// Root cause: the `Sema::ValidateBoundsAttrTypeShape` checks
// runs inside `ConstructDynamicBoundType::Visit()` once per peeled layer of
// sugar. For an unknown-size pointee it emits the error and then *recovers*
// (sets `Flags.CountInBytes` and returns true), so the walk continues.
// `VisitTypedefType` (via `HandleNamedAliasType`) re-enters `Visit()` on the
// desugared pointer at the same level, so the check runs a second time -- and
// the function/sizeless-pointee branch is not gated on `Flags.CountInBytes`, so
// it re-fires (now spelled `sized_by`, because `CountInBytes` was flipped).
//
// The `isa<AttributedType>` guard in `ConstructDynamicBoundType::Visit()`
// suppresses this for `AttributedType` layers, but not for the "named alias"
// family (`typedef` / `using` / `__typeof__` / `decltype`). It reproduces only
// in attribute-only mode; full `-fbounds-safety` replaces the typedef with an
// implicit `__single` `AttributedType`, which the guard does cover.

#include <ptrcheck.h>

typedef int fn_t(int);
typedef fn_t *fnptr_t;

int len;

// Bare pointer: a single diagnostic (correct).
// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'fn_t' (aka 'int (int)') is a function type}}
fn_t *bare __counted_by(len);

// FIXME:
// Through a typedef: the diagnostic is emitted twice; the second (`sized_by`)
// expected-error@+2{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'fn_t' (aka 'int (int)') is a function type}}
// expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'fn_t' (aka 'int (int)') is a function type}}
fnptr_t alias __counted_by(len);
