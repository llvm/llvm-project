// RUN: %clang_analyze_cc1 -verify %s -fcxx-exceptions -fexceptions -analyzer-checker=core,alpha.deadcode.UnreachableCode

// expected-no-diagnostics

void foo();

void fp_90162() {
  try { // no-warning: The TryStmt shouldn't be unreachable.
    foo();
  } catch (int) {
    foo(); // We assume that catch handlers are reachable.
  }
}
