// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
//
// This test guards a precision fix in RangedConstraintManager.  Previously the
// assume machinery simplified a symbol to an SVal but *discarded* the result
// unless it was still a SymbolRef.  When a symbol simplifies to a concrete
// integer (e.g. the folded value of `(x & 137) & 8` once `(x & 137)` is pinned
// to 2), the concrete was thrown away and the coarse, over-approximated range
// of the original symbol was used instead, which kept a self-contradictory
// path feasible.  The assumeSym* entry points now decide feasibility directly
// from the concrete simplification (via simplifyToSVal), so the dead paths
// below are correctly pruned.
//
// Three assume entry points consumed the discarded simplification; each is
// exercised below.

void clang_analyzer_warnIfReached(void);
void clang_analyzer_eval(int);

long global_var;

// (1) RangedConstraintManager::assumeSymUnsupported.
//
// A bare bitwise expression used as a branch condition is not a comparison, so
// canReasonAbout() returns false and SimpleConstraintManager routes it to
// assumeSymUnsupported() -> assumeSymNE(sym, 0).  The symbol folds to the
// concrete 0; previously that fold was discarded and assumeSymNE deleted the
// point 0 from the *coarse* range [0, 2] of `(global_var & 137) & 8`, leaving
// [1, 2], which looked feasible and (wrongly) entered the dead branch.  The
// concrete fold is now honored: 0 != 0 is false, so the branch is pruned.
void assumeSymUnsupported_bitwise(void) {
  if ((global_var & 137) == 2)
    if ((global_var & 137) & 8)
      clang_analyzer_warnIfReached(); // no-warning (dead: (2 & 8) == 0)
}

// (2) RangedConstraintManager::assumeSymInclusiveRange.
//
// A switch over a bitwise expression with a GNU case-range routes through
// ExprEngine's assumeInclusiveRange() -> assumeSymInclusiveRange().  Same
// story: the switch value folds to the concrete 0.  0 is not in [1, 100], so
// the case is dead; honoring the concrete fold prunes it.
void assumeSymInclusiveRange_switch(void) {
  if ((global_var & 137) == 2)
    switch ((global_var & 137) & 8) {
    case 1 ... 100:
      clang_analyzer_warnIfReached(); // no-warning (dead: 0 not in [1,100])
      break;
    }
}

// (3) RangedConstraintManager::assumeSym.
//
// assumeSym() is only reached for conditions canReasonAbout() accepts, i.e.
// comparison (or +/-) symbolic expressions.  Unlike the two cases above, the
// discarded concrete here was already masked: an explicit comparison is folded
// by simplifySVal() inside evalBinOp() at evaluation time (or collapses the
// operand to a single concrete value, which triggers the assignSymExprToConst
// cascade), so the contradiction was detected and the branch pruned regardless.
// The evals below document that the analyzer knows the value is 0 on this path.
void assumeSym_comparison(void) {
  if ((global_var & 137) == 2) {
    clang_analyzer_eval(((global_var & 137) & 8) == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(((global_var & 137) & 8) != 0); // expected-warning{{FALSE}}
    // Consequently the comparison branch is correctly pruned (no warning):
    if (((global_var & 137) & 8) > 0)
      clang_analyzer_warnIfReached(); // no-warning (correctly unreachable)
  }
}
