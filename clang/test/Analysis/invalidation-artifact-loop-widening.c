// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config widen-loops=true -analyzer-max-loop 2 \
// RUN:   -verify %s

void clang_analyzer_dump(int);

// When a loop is widened, the analyzer invalidates stack locals, stack
// arguments, and globals. After this patch the values produced by that
// invalidation are SymbolInvalidationArtifact carrying a "loop-widening"
// cause, instead of plain SymbolConjured.

void test_widening_marks_stack_local(void) {
  int x = 0;
  for (int i = 0; i < 1000; ++i) {
    x = i;
  }
  // After widening, x is bound to a SymbolInvalidationArtifact whose cause is "loop-widening".
  clang_analyzer_dump(x); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, loop-widening, S[0-9]+, #[0-9]+}}}}}
}
