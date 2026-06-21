// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DebugLifetimeAnnotations \
// RUN:   -verify %s

void clang_analyzer_lifetime_bound(int);

// Verify that the DebugLifetimeAnnotations checkre can be used without
// the LifetimeAnnotations checker being enabled and it does not cause
// crash.
void test() {
  int x = 5;
  clang_analyzer_lifetime_bound(x); // expected-no-diagnostics
}
