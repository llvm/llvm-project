// RUN: %clang_analyze_cc1 -analyzer-output=plist -o %t.plist -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/plist-diagnostics.cpp.plist -

void log();

int max(int a, int b) { // expected-warning{{Duplicate code detected}}
  log();
  if (a > b)
    return a;
  return b;
}

int maxClone(int a, int b) { // expected-note{{Similar code here}}
  log();
  if (a > b)
    return a;
  return b;
}

