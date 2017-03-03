// RUN: %clang_analyze_cc1 -fcxx-exceptions -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

bool foo1(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    try { x--; } catch (int i) {}
  return true;
}

// Uses parenthesis instead of type
bool foo2(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    try { x--; } catch (...) {}
  return true;
}

// Catches a different type (long instead of int)
bool foo3(int x) {
  if (x > 0)
    return false;
  else if (x < 0)
    try { x--; } catch (long i) {}
  return true;
}
