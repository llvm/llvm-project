// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:  -analyzer-constraints=z3 

// REQUIRES: Z3
//
// Previously Z3 analysis crashed when it encountered an UnarySymExpr, validate
// that this no longer happens.
//

// expected-no-diagnostics
int negate(int x, int y) {
  if ( ~(x && y))
    return 0;
  return 1;
}
