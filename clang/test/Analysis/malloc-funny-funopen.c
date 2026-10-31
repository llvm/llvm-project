// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Malloc -verify %s
#include "Inputs/system-header-simulator-funny-funopen.h"

// expected-no-diagnostics

// Standard funopen() has 5 parameters; but if a system header contained a
// nonstandard declaration with only 4 parameters, then calling that used to
// trigger an assertion failure through MallocChecker.
void use_funny_funopen(int *a, int *b, int *c, int *d) {
  funopen(a, b, c, d); // no-crash
}
