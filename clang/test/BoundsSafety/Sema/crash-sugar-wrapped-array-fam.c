// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// Regression test for a compiler crash (reachable assertion) in
// ConstructDynamicBoundType::VisitType when the incomplete-array type of a
// flexible array member is hidden behind sugar (a typedef or __typeof__).
//

#include <ptrcheck.h>

// expected-no-diagnostics

// Baseline: a plain FAM with __counted_by is valid.
struct Plain {
  int n;
  int fam[] __counted_by(n);
};

// These variants below were previously disallowed by or crashed older Clang
// compilers.


// The incomplete-array type comes from a typedef.
typedef int int_array_t[];
struct ViaTypedef {
  int n;
  int_array_t fam __counted_by(n);
};

// The incomplete-array type comes from __typeof__ of an incomplete-array
// expression (a TypeOfExprType, distinct from a typedef).
extern int global_array[];
struct ViaTypeof {
  int n;
  __typeof__(global_array) fam __counted_by(n);
};

// The incomplete-array type is reached through several nested typedefs.
typedef int_array_t int_array2_t;
typedef int_array2_t int_array3_t;
struct ViaNestedTypedef {
  int n;
  int_array3_t fam __counted_by(n);
};
