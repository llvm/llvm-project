// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Applying a count/size/range attribute to a __terminated_by pointer is
// rejected, including when the __terminated_by pointer is reached through a
// typedef. ValidateBoundsAttrTypeShape walks sugar to find the
// ValueTerminatedType behind the typedef, so the conflict is diagnosed. A
// top-node-only check would miss the typedef and silently build the attribute
// over the terminated pointer.
//
// The typedef is written with an explicit `__single` so that -fbounds-safety
// auto-bounding leaves the underlying type unchanged and therefore retains the
// typedef (see MakeAutoPointer::VisitTypedefType). Without the explicit
// `__single`, auto-bounding rewrites the pointer and drops the typedef, so the
// type reaching the check is already a bare ValueTerminatedType and the
// sugar-walking is not exercised.
//
// The count/end sibling is declared before the attributed field so the
// attribute is resolved on the eager path (ConstructDynamicBoundType), where
// this check runs.

#include <ptrcheck.h>

typedef int *__single __null_terminated nt_ptr_t;
typedef nt_ptr_t nt_ptr2_t;

// FIXME: These diagnostics are incredibly misleading (rdar://185163524).

struct counted {
  int n;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr_t p __counted_by(n);
};

struct counted_or_null {
  int n;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr_t p __counted_by_or_null(n);
};

struct sized {
  int n;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr_t p __sized_by(n);
};

struct sized_or_null {
  int n;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr_t p __sized_by_or_null(n);
};

struct ended {
  int *e;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr_t p __ended_by(e);
};

// Through a chain of typedefs -- still diagnosed exactly once.
struct counted_chain {
  int n;
  // expected-error@+1{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
  nt_ptr2_t p __counted_by(n);
};
