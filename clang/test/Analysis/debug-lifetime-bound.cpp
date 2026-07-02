// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UseAfterLifetimeEnd,debug.DebugLifetimeModeling -verify %s

// expected-no-diagnostics

void clang_analyzer_dumpLifetimeOriginsOf(int);

void test() {
  int x = 5;
  clang_analyzer_dumpLifetimeOriginsOf(x); // no-warning
}

