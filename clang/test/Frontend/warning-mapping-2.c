// Check that -w takes precedence over -pedantic-errors.
// RUN: %clang_cc1 -verify -pedantic-errors -w %s

// Expect *not* to see a diagnostic for "implicit declaration of function"
// expected-no-diagnostics

void f0(void) { f1(); }
