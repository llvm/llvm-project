// RUN: %clang_analyze_cc1 -std=c11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// expected-no-diagnostics

int global;

int foo1() {
  if (global > 0)
    return 0;
  else if (global < 0)
    return _Generic(global, double: 1, float: 2, default: 3);
  return 1;
}

// Different associated type (int instead of float)
int foo2() {
  if (global > 0)
    return 0;
  else if (global < 0)
    return _Generic(global, double: 1, int: 2, default: 4);
  return 1;
}

// Different number of associated types.
int foo3() {
  if (global > 0)
    return 0;
  else if (global < 0)
    return _Generic(global, double: 1, default: 4);
  return 1;
}
