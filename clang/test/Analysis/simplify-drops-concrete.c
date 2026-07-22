// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
//
// This test documents an imprecision in RangedConstraintManager: the helper
//
//   SymbolRef simplify(ProgramStateRef, SymbolRef)
//
// (clang/lib/StaticAnalyzer/Core/RangedConstraintManager.cpp) simplifies a
// symbol to an SVal but then *discards* the result unless it is still a
// SymbolRef.  When a symbol simplifies to a concrete integer (e.g. the folded
// value of `(x & 137) & 8` once `(x & 137)` is pinned to 2), the concrete is
// thrown away and the coarse, over-approximated range of the original symbol
// is used instead.  That keeps a self-contradictory path feasible.
//
// The helper is called from three places; each is exercised below.

void clang_analyzer_warnIfReached(void);
void clang_analyzer_eval(int);

long global_var;

// (1) simplify() call inside RangedConstraintManager::assumeSymUnsupported.
//
// A bare bitwise expression used as a branch condition is not a comparison, so
// canReasonAbout() returns false and SimpleConstraintManager routes it to
// assumeSymUnsupported() -> assumeSymNE(sym, 0).  simplify() folds the symbol
// to the concrete 0 but discards it; assumeSymNE then deletes the point 0 from
// the *coarse* range [0, 2] of `(global_var & 137) & 8`, leaving [1, 2], which
// looks feasible.  The dead branch is therefore (incorrectly) entered.
//
// FIXME: This branch is dead ((2 & 8) == 0), so it should NOT be reachable.
// The REACHABLE expectation below encodes the current (buggy) behavior and
// should be removed once simplify() preserves concrete simplifications.
void assumeSymUnsupported_bitwise(void) {
  if ((global_var & 137) == 2)
    if ((global_var & 137) & 8)
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

// (2) simplify() call inside RangedConstraintManager::assumeSymInclusiveRange.
//
// A switch over a bitwise expression with a GNU case-range routes through
// ExprEngine's assumeInclusiveRange() -> assumeSymInclusiveRange().  Same
// story: simplify() folds the switch value to 0 but discards it, and the
// coarse range [0, 2] intersected with the case range [1, 100] yields the
// non-empty [1, 2], so the (dead) case is entered.
//
// FIXME: 0 is not in [1, 100], so this case is dead and should be unreachable.
void assumeSymInclusiveRange_switch(void) {
  if ((global_var & 137) == 2)
    switch ((global_var & 137) & 8) {
    case 1 ... 100:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    }
}

// (3) simplify() call inside RangedConstraintManager::assumeSym.
//
// assumeSym() is only reached for conditions canReasonAbout() accepts, i.e.
// comparison (or +/-) symbolic expressions.  Unlike the two cases above, the
// discarded concrete here is masked: an explicit comparison is folded by
// simplifySVal() inside evalBinOp() at evaluation time (or collapses the
// operand to a single concrete value, which triggers the assignSymExprToConst
// cascade), so the contradiction is detected and the branch is pruned
// regardless of what simplify() returns.  We therefore cannot demonstrate a
// spurious *reachable* branch through assumeSym; the discard here is only
// wasted work.  The evals below document that the analyzer already knows the
// value is 0 on this path.
void assumeSym_comparison(void) {
  if ((global_var & 137) == 2) {
    clang_analyzer_eval(((global_var & 137) & 8) == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(((global_var & 137) & 8) != 0); // expected-warning{{FALSE}}
    // Consequently the comparison branch is correctly pruned (no warning):
    if (((global_var & 137) & 8) > 0)
      clang_analyzer_warnIfReached(); // no-warning (correctly unreachable)
  }
}
