// RUN: %clang_cc1 -std=c99 -analyze -analyzer-checker=core,alpha.core.BoolAssignment -analyzer-store=region -verify %s

// Test stdbool.h's _Bool

// Prior to C99, stdbool.h uses this typedef, but even in ANSI C mode, _Bool
// appears to be defined.

// #if __STDC_VERSION__ < 199901L
// typedef int _Bool;
// #endif

void test_stdbool_initialization(int y) {
  if (y < 0) {
    _Bool x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    _Bool x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  _Bool x = y; // no-warning
}

void test_stdbool_assignment(int y) {
  _Bool x = 0; // no-warning
  if (y < 0) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  x = y; // no-warning
}
