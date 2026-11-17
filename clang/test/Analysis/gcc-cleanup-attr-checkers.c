// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,cplusplus,security,deadcode,nullability,optin.portability,optin.performance,optin.core,debug.ExprInspection -verify %s

// Run the cleanup function modeling with a broad set of checkers: the
// CleanupFunctionCall has an argument without a source expression and a
// result type that does not describe the cleanup function's return value,
// which must not crash checkers that inspect call arguments or results.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_dump_ptr(void *);
void clang_analyzer_warnIfReached(void);

void first_entry_point(void) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

void declared_only_cleanup(void *p);

// The cleanup function has a non-void return type; its return value is
// ignored, and CallEvent::getResultType() reports void for the implicit call.
static int non_void_cleanup(int *p) {
  (void)p;
  return 42;
}

static void noop_cleanup(int *p) {
  clang_analyzer_dump_ptr(p); // expected-warning {{&x}}
  (void)p;
}

void many_checkers_with_cleanup(void) {
  int y __attribute__((cleanup(non_void_cleanup)));
  void *p __attribute__((cleanup(declared_only_cleanup)));
  {
    int x __attribute__((cleanup(noop_cleanup)));
    x = 42; // no dead-store warning: the value is read by the cleanup call.
  }
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  y = 42;
  p = malloc(10);
} // no leak: the memory escapes into the conservatively evaluated call.

void last_entry_point(void) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}
