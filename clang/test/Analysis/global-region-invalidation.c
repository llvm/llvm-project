// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -disable-free -analyzer-eagerly-assume -analyzer-checker=core,deadcode,alpha.security.taint,debug.TaintTest,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

// Note, we do need to include headers here, since the analyzer checks if the function declaration is located in a system header.
#include "Inputs/system-header-simulator.h"

// Test that system header does not invalidate the internal global.
int size_rdar9373039 = 1;
int rdar9373039() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;

  // strlen doesn't invalidate the value of 'size_rdar9373039'.
  int extra = (2 + strlen ("Clang") + ((4 - ((unsigned int) (2 + strlen ("Clang")) % 4)) % 4)) + (2 + strlen ("1.0") + ((4 - ((unsigned int) (2 + strlen ("1.0")) % 4)) % 4));

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // no-warning

  return j;
}

// Test stdin does not get invalidated by a system call nor by an internal call.
void foo();
int stdinTest() {
  int i = 0;
  fscanf(stdin, "%d", &i);
  foo();
  int m = i; // expected-warning + {{tainted}}
  fscanf(stdin, "%d", &i);
  int j = i; // expected-warning + {{tainted}}
  return m + j; // expected-warning + {{tainted}}
}

// Test errno gets invalidated by a system call.
int testErrnoSystem() {
  int i;
  int *p = 0;
  fscanf(stdin, "%d", &i);
  if (errno == 0) {
    fscanf(stdin, "%d", &i); // errno gets invalidated here.
    return 5 / errno; // no-warning
  }
  return 0;
}

// Test that errno gets invalidated by internal calls.
int testErrnoInternal() {
  int i;
  int *p = 0;
  fscanf(stdin, "%d", &i);
  if (errno == 0) {
    foo(); // errno gets invalidated here.
    return 5 / errno; // no-warning
  }
  return 0;
}

// Test that const integer does not get invalidated.
const int x = 0;
int constIntGlob() {
  const int *m = &x;
    foo();
  return 3 / *m; // expected-warning {{Division by zero}}
}

extern const int x;
int constIntGlobExtern() {
  if (x == 0) {
    foo();
    return 5 / x; // expected-warning {{Division by zero}}
  }
  return 0;
}

void testAnalyzerEvalIsPure() {
  extern int someGlobal;
  if (someGlobal == 0) {
    clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
  }
}

