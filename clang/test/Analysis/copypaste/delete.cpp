// RUN: %clang_analyze_cc1 -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// expected-no-diagnostics

bool foo1(int x, int* a) {
  if (x > 0)
    return false;
  else if (x < 0)
    delete a;
  return true;
}

// Explicit global delete
bool foo2(int x, int* a) {
  if (x > 0)
    return false;
  else if (x < 0)
    ::delete a;
  return true;
}

// Array delete
bool foo3(int x, int* a) {
  if (x > 0)
    return false;
  else if (x < 0)
    delete[] a;
  return true;
}
