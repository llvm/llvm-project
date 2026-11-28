// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,cplusplus,security,deadcode,nullability,optin.portability,optin.performance,optin.core,debug.ExprInspection -verify %s

// CleanupFunctionCall has no source expression for its argument and reports a
// void result; run a broad checker set to ensure no checker crashes on it.

#include "Inputs/system-header-simulator-for-malloc.h"

void clang_analyzer_warnIfReached(void);

void declared_only_cleanup(void *p);

// The cleanup function has a non-void return type; its return value is
// ignored, and CallEvent::getResultType() reports void for the implicit call.
static int non_void_cleanup(int *p) {
  (void)p;
  return 42;
}

void many_checkers_with_cleanup(void) {
  int y __attribute__((cleanup(non_void_cleanup)));
  void *p __attribute__((cleanup(declared_only_cleanup)));
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  y = 42;
  p = malloc(10);
} // No leak: the memory escapes into the conservatively evaluated call.
