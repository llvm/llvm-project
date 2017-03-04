// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s 2>&1 | FileCheck %s

// Self-tests for the debug.ExprInspection checker.

void clang_analyzer_dump(int x);
void clang_analyzer_printState();
void clang_analyzer_numTimesReached();

void foo(int x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<int x>}}
  int y = 1;
  clang_analyzer_printState();
  for (; y < 3; ++y)
    clang_analyzer_numTimesReached(); // expected-warning{{2}}
}

// CHECK: Store (direct and default bindings)
// CHECK-NEXT: (y,0,direct) : 1 S32b

// CHECK: Expressions:
// CHECK-NEXT: clang_analyzer_printState : &code{clang_analyzer_printState}
// CHECK-NEXT: Ranges are empty.
