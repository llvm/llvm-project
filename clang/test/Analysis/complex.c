// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -Wno-unreachable-code -ffreestanding \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection

#include <stdint.h>

int clang_analyzer_eval(int);

void f1(int * p) {
  // This branch should be infeasible
  // because __imag__ p is 0.
  if (!p && __imag__ (intptr_t) p)
    *p = 1; // no-warning

  // If p != 0 then this branch is feasible; otherwise it is not.
  if (__real__ (intptr_t) p)
    *p = 1; // no-warning
    
  *p = 2; // expected-warning{{Dereference of null pointer}}
}

void complexFloat(__complex__ float f) {
  clang_analyzer_eval(__real__(f) == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(__imag__(f) == 1); // expected-warning{{UNKNOWN}}

  __real__(f) = 1;
  __imag__(f) = 1;

  clang_analyzer_eval(__real__(f) == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(__imag__(f) == 1); // expected-warning{{UNKNOWN}}
}

void complexInt(__complex__ int f) {
  clang_analyzer_eval(__real__(f) == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(__imag__(f) == 1); // expected-warning{{UNKNOWN}}

  __real__(f) = 1;
  __imag__(f) = 1;

  clang_analyzer_eval(__real__(f) == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(__imag__(f) == 1); // expected-warning{{UNKNOWN}}
}
