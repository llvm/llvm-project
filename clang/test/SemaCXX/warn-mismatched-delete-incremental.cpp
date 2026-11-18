// RUN: %clang_cc1 -fincremental-extensions -verify %s
// Regression test for warn_mismatched_delete_new in incremental mode
// in Sema::ActOnEndOfTranslationUnit(). Ensures the diagnostic fires correctly
// in incremental mode and that DeleteExprs does not accumulate stale entries
// across EndOfTU cycles.

struct S {
  int *p;
  S() : p(new int[10]) {}
  ~S() { delete p; } // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
                      // expected-note@-2 {{allocated with 'new[]' here}}
};