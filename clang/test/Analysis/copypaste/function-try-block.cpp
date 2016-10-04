// RUN: %clang_cc1 -analyze -fcxx-exceptions -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// Tests if function try blocks are correctly handled.

void nonCompoundStmt1(int& x)
  try { x += 1; } catch(...) { x -= 1; } // expected-warning{{Detected code clone.}}

void nonCompoundStmt2(int& x)
  try { x += 1; } catch(...) { x -= 1; } // expected-note{{Related code clone is here.}}
