// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -disable-free -analyzer-checker=core,deadcode,alpha.security.taint,debug.TaintTest,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

// Note, we do need to include headers here, since the analyzer checks if the function declaration is located in a system header.
#include "Inputs/system-header-simulator.h"

// Test that system header does not invalidate the internal global.
int size_rdar9373039 = 1;
int rdar9373039(void) {
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
void foo(void);
int stdinTest(void) {
  int i = 0;
  fscanf(stdin, "%d", &i);
  foo();
  int m = i; // expected-warning + {{tainted}}
  fscanf(stdin, "%d", &i);
  int j = i; // expected-warning + {{tainted}}
  return m + j; // expected-warning + {{tainted}}
}

// Test errno gets invalidated by a system call.
int testErrnoSystem(void) {
  int i;
  int *p = 0;
  fscanf(stdin, "%d", &i);
  if (errno == 0) {
    fscanf(stdin, "%d", &i); // errno gets invalidated here.
    return 5 / errno; // no-warning
  }

  errno = 0;
  fscanf(stdin, "%d", &i); // errno gets invalidated here.
  return 5 / errno; // no-warning
}

// Test that errno gets invalidated by internal calls.
int testErrnoInternal(void) {
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
int constIntGlob(void) {
  const int *m = &x;
    foo();
  return 3 / *m; // expected-warning {{Division by zero}}
}

extern const int y;
int constIntGlobExtern(void) {
  if (y == 0) {
    foo();
    return 5 / y; // expected-warning {{Division by zero}}
  }
  return 0;
}

static void * const ptr = 0;
void constPtrGlob(void) {
  clang_analyzer_eval(ptr == 0); // expected-warning{{TRUE}}
  foo();
  clang_analyzer_eval(ptr == 0); // expected-warning{{TRUE}}
}

static const int x2 = x;
void constIntGlob2(void) {
  clang_analyzer_eval(x2 == 0); // expected-warning{{TRUE}}
  foo();
  clang_analyzer_eval(x2 == 0); // expected-warning{{TRUE}}
}

void testAnalyzerEvalIsPure(void) {
  extern int someGlobal;
  if (someGlobal == 0) {
    clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(someGlobal == 0); // expected-warning{{TRUE}}
  }
}

// Test that static variables with initializers do not get reinitialized on
// recursive calls.
void Function2(void);
int *getPtr(void);
void Function1(void) {
  static unsigned flag;
  static int *p = 0;
  if (!flag) {
    flag = 1;
    p = getPtr();
  }
  int m = *p; // no-warning: p is never null.
  m++;
  Function2();
}
void Function2(void) {
    Function1();
}

void SetToNonZero(void) {
  static int g = 5;
  clang_analyzer_eval(g == 5); // expected-warning{{TRUE}}
}

