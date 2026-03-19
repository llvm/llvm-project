// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// ============================================================================
// Group A: Compound statement (block)
// ============================================================================

void suppress_compound_suppressed() {
  [[clang::suppress]] {
    clang_analyzer_warnIfReached(); // no-warning
  }
}

void suppress_compound_unsuppressed() {
  {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group B: If statement
// ============================================================================

void suppress_if(bool coin) {
  [[clang::suppress]] if (coin) {
    clang_analyzer_warnIfReached(); // no-warning
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void suppress_if_else(bool coin) {
  [[clang::suppress]] if (coin) {
    clang_analyzer_warnIfReached(); // no-warning
  } else {
    clang_analyzer_warnIfReached(); // no-warning: entire if-else is suppressed
  }
}

void unsuppressed_if_else(bool coin) {
  if (coin) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  } else {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group C: Loop statements
// ============================================================================

void suppress_for(int n) {
  [[clang::suppress]] for (int i = 0; i < n; ++i) {
    clang_analyzer_warnIfReached(); // no-warning
  }
}

void suppress_while(int n) {
  [[clang::suppress]] while (--n) {
    clang_analyzer_warnIfReached(); // no-warning
  }
}

void suppress_do_while(int n) {
  [[clang::suppress]] do {
    clang_analyzer_warnIfReached(); // no-warning
  } while (--n);
}

void unsuppressed_for(int n) {
  for (int i = 0; i < n; ++i) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

void suppress_range_for() {
  int arr[] = {1, 2, 3};
  [[clang::suppress]] for (int x : arr) {
    clang_analyzer_warnIfReached(); // no-warning
    (void)x;
  }
}

// ============================================================================
// Group D: Switch statement
// ============================================================================

int suppress_switch(int n) {
  [[clang::suppress]] switch (n) {
  case 1:
    return clang_analyzer_warnIfReached(), 1; // no-warning
  default:
    break;
  }
  return 0;
}

int unsuppressed_switch(int n) {
  switch (n) {
  case 1:
    return clang_analyzer_warnIfReached(), 1; // expected-warning{{REACHABLE}}
  default:
    break;
  }
  return 0;
}

// ============================================================================
// Group E: Return statement
// ============================================================================

int suppress_return() {
  [[clang::suppress]] return clang_analyzer_warnIfReached(), 1; // no-warning
}

int unsuppressed_return() {
  return clang_analyzer_warnIfReached(), 1; // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group F: Expression statement
// ============================================================================

void suppress_expr_stmt() {
  [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning
}

void unsuppressed_expr_stmt() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group G: Nested suppressed blocks
// ============================================================================

void nested_suppression() {
  [[clang::suppress]] {
    [[clang::suppress]] {
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
}

// ============================================================================
// Group H: Suppression on single statement within method
// ============================================================================

struct H_ClassWithMethods {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_H() {
  H_ClassWithMethods().method();
}
