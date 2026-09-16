// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

#include "Inputs/std-c-library-functions-POSIX.h"
#define NULL ((void*)0)

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();


// An opaque function that returns a symbolic pointer in unknown space.
int *opaque_function(int *p);

void test_simple(void) {
  // Validate that the analyzer doesn't rule out the equality (or disequality)
  // of a pointer to the stack ('&x') and a symbolic pointer in unknown space.
  // Previously the analyzer incorrectly assumed that stack pointers cannot be
  // equal to symbolic pointers, which is obviously nonsense. It is true that
  // functions cannot validly return pointers to their own stack frame, but
  // they can easily return a pointer to some other stack frame (e.g. in this
  // example 'opaque_function' could return its argument).
  int x = 0;
  if (&x == opaque_function(&x)) {
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  } else {
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  }
}

void test_local_not_leaked(void) {
  // In this situation a very smart analyzer could theoretically deduce that
  // the address of the local 'x' cannot leak from this function, so the
  // call to 'opaque_function' cannot return it.
  int x = 0;
  if (&x == opaque_function(NULL)) {
    // This branch is unreachable (without non-standard-compliant tricks);
    // however, we cannot blame the analyzer for not deducing this.
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  } else {
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  }
}

void test_fgets_can_succeed(FILE *infile) {
  // The modeling of 'fgets' in StdCLibraryFunctions splits two branches: one
  // where the return value is assumed to be equal to the first argument, and
  // one where the return value is assumed to be null. However, if the target
  // buffer was on the stack, then 'evalBinOp' rejected the possibility that
  // the return value (a symbolic pointer) can be equal to the first argument
  // (a pointer to the stack), so the analyzer was unable to enter the
  // "success" branch.
  char buffer[100];
  if (fgets(buffer, 100, infile) != NULL) {
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  } else {
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  }
}
