// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection %s -verify

// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ConfigDumper 2>&1 | FileCheck %s --match-full-lines
// CHECK: max-symbol-complexity = 35

void clang_analyzer_dump(int v);

void pumpSymbolComplexity() {
  extern int *p;
  *p = (*p + 1) & 1023; //  2
  *p = (*p + 1) & 1023; //  4
  *p = (*p + 1) & 1023; //  6
  *p = (*p + 1) & 1023; //  8
  *p = (*p + 1) & 1023; // 10
  *p = (*p + 1) & 1023; // 12
  *p = (*p + 1) & 1023; // 14
  *p = (*p + 1) & 1023; // 16
  *p = (*p + 1) & 1023; // 18
  *p = (*p + 1) & 1023; // 20
  *p = (*p + 1) & 1023; // 22
  *p = (*p + 1) & 1023; // 24
  *p = (*p + 1) & 1023; // 26
  *p = (*p + 1) & 1023; // 28
  *p = (*p + 1) & 1023; // 30
  *p = (*p + 1) & 1023; // 32
  *p = (*p + 1) & 1023; // 34

  // The complexity of "*p" is below 35, so it's accurate.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^\({34}reg}}}}

  // We would increase the complexity over the threshold, thus it'll get simplified.
  *p = (*p + 1) & 1023; // Would be 36, which is over 35.

  // This dump used to print a hugely complicated symbol, over 800 complexity, taking really long to simplify.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^}}(complex_${{[0-9]+}}) & 1023}} [debug.ExprInspection]{{$}}}}
}

void hugelyOverComplicatedSymbol() {
#define TEN_TIMES(x) x x x x x x x x x x
#define HUNDRED_TIMES(x) TEN_TIMES(TEN_TIMES(x))
  extern int *p;
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  *p = (*p + 1) & 1023;
  *p = (*p + 1) & 1023;
  *p = (*p + 1) & 1023;
  *p = (*p + 1) & 1023;

  // This dump used to print a hugely complicated symbol, over 800 complexity, taking really long to simplify.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^}}(((complex_${{[0-9]+}}) & 1023) + 1) & 1023 [debug.ExprInspection]{{$}}}}
#undef HUNDRED_TIMES
#undef TEN_TIMES
}
