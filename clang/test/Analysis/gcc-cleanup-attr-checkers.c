// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,cplusplus,security,deadcode,nullability,optin.portability,optin.performance,optin.core,debug.ExprInspection -verify %s

// Run the cleanup function modeling with a broad set of checkers: the
// CleanupFunctionCall has an argument without a source expression, which
// must not crash checkers that inspect call arguments.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_warnIfReached(void);

void declared_only_cleanup(void *p);

static void noop_cleanup(int *p) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  (void)p;
}

void many_checkers_with_cleanup(void) {
  int x __attribute__((cleanup(noop_cleanup)));
  x = 42; // no dead-store warning: the value is read by the cleanup call.
  void *p __attribute__((cleanup(declared_only_cleanup)));
  p = malloc(10);
} // no leak: the memory escapes into the conservatively evaluated call.

void analysis_continues_after_cleanup(void) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}
