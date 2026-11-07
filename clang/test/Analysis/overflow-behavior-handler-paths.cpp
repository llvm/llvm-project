// RUN: %clang_analyze_cc1 -std=c++23 -fexperimental-overflow-behavior-types -triple x86_64-unknown-linux-gnu -analyzer-checker=core,debug.ExprInspection -verify %s

// Paths through an overflow handler that the analyzer must survive.

void clang_analyzer_warnIfReached();

// The right operand of a logical operator is the last statement on its path.
int logical_rhs(int __ob_trap a, int r) {
  return r && (a + 3);
__overflow:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return -1;
}

// Checks in a new-expression's initializer trap, so no path leaves a
// construction unfinished.
struct S {
  S(int);
};
S *new_init(int __ob_trap a) {
  return new S(a + 1);
__overflow:
  return nullptr;
}

// The array size is computed before the allocation and may branch.
int *new_size(int __ob_trap n) {
  return new int[n * 2];
__overflow:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return nullptr;
}
