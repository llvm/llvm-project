// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=true -DSUPPRESSED=1 -verify %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

#include "../Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

void testCopyNull(int *I, int *E) {
  std::copy(I, E, (int *)0);
#ifndef SUPPRESSED
  // expected-warning@../Inputs/system-header-simulator-cxx.h:110 {{Dereference of null pointer}}
#endif
}
