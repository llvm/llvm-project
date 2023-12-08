// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// This tests if we search for clones in functions.

void log();

int max(int a, int b) { // expected-warning{{Duplicate code detected}}
  log();
  if (a > b)
    return a;
  return b;
}

int maxClone(int x, int y) { // expected-note{{Similar code here}}
  log();
  if (x > y)
    return x;
  return y;
}

// Functions below are not clones and should not be reported.

// The next two functions test that statement classes are still respected when
// checking for clones in expressions. This will show that the statement
// specific data of all base classes is collected, and not just the data of the
// first base class.
int testBaseClass(int a, int b) { // no-warning
  log();
  if (a > b)
    return true ? a : b;
  return b;
}
int testBaseClass2(int a, int b) { // no-warning
  log();
  if (a > b)
    return __builtin_choose_expr(true, a, b);
  return b;
}

// No clone because of the different comparison operator.
int min1(int a, int b) { // no-warning
  log();
  if (a < b)
    return a;
  return b;
}

// No clone because of the different pattern in which the variables are used.
int min2(int a, int b) { // no-warning
  log();
  if (a > b)
    return b;
  return a;
}

int foo(int a, int b) { // no-warning
  return a + b;
}
