// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
#define CHECK(expr) if (!(expr)) return; clang_analyzer_eval(expr)

void testPersistentConstraints(int x, int y) {
  // Sanity check
  CHECK(x); // expected-warning{{TRUE}}
  CHECK(x & 1); // expected-warning{{TRUE}}
  
  // False positives due to SValBuilder giving up on certain kinds of exprs.
  CHECK(1 - x); // expected-warning{{UNKNOWN}}
  CHECK(x & y); // expected-warning{{UNKNOWN}}
}