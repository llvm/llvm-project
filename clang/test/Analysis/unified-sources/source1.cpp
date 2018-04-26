// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// This test tests that the warning is here when it is included from
// the unified sources file. The run-line in this file is there
// only to suppress LIT warning for the complete lack of run-line.
int foo(int x) {
  if (x) {}
  return 1 / x; // expected-warning{{}}
}

// Let's see if the container inlining heuristic still works.
#include "container.h"
int testContainerMethodInHeaderFile(ContainerInHeaderFile Cont) {
  return 1 / Cont.method(); // no-warning
}
