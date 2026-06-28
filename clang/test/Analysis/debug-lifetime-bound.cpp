// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.LifetimeAnnotations,debug.DebugLifetimeAnnotations -verify %s

// expected-no-diagnostics

void clang_analyzer_lifetime_bound(int);

void test() {
  int x = 5;
  clang_analyzer_lifetime_bound(x); // no-warning: verifies debug checker does not crash standalone
}
