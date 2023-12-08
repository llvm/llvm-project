// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist -o %t %s
// RUN: %normalize_plist <%t | diff -ub %S/Inputs/expected-plists/undef-value-caller.c.plist -

#include "undef-value-callee.h"

// This code used to cause a crash since we were not adding fileID of the header to the plist diagnostic.

int test_calling_unimportant_callee(int argc, char *argv[]) {
  int x;
  callee();
  return x; // expected-warning {{Undefined or garbage value returned to caller}}
}

