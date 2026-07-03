// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump(int);

void test_atomic_load_invalidates_target(void) {
  int x = 42;
  int loaded;
  // The atomic load is conservatively modeled as invalidating its argument.
  __c11_atomic_load((_Atomic int *)&x, 0);
  clang_analyzer_dump(x); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, unmodeled-expr AtomicExpr, S[0-9]+, #[0-9]+}}}}}
  (void)loaded;
}

void test_inline_asm_invalidates_outputs(void) {
  int x = 0;
  asm("nop" : "=r"(x)); // GCC inline asm invalidates its output operand.
  clang_analyzer_dump(x); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, unmodeled-expr GCCAsmStmt, S[0-9]+, #[0-9]+}}}}}
}
