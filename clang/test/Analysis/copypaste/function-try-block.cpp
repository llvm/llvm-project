// RUN: %clang_analyze_cc1 -fcxx-exceptions -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=10 -verify %s

// Tests if function try blocks are correctly handled.

void nonCompoundStmt1(int& x)
  try { x += 1; } catch(...) { x -= 1; } // expected-warning{{Duplicate code detected}}

void nonCompoundStmt2(int& x)
  try { x += 1; } catch(...) { x -= 1; } // expected-note{{Similar code here}}
