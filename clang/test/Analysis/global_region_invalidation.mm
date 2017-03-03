// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

#include "Inputs/system-header-simulator.h"

void use(int);
id foo(int x) {
  if (x)
    return 0;
  static id p = foo(1); 
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  return p;
}

const int &globalIntRef = 42;

void testGlobalRef() {
  // FIXME: Should be TRUE, but should at least not crash.
  clang_analyzer_eval(globalIntRef == 42); // expected-warning{{UNKNOWN}}
}

extern int globalInt;
struct IntWrapper {
  int value;
};
extern struct IntWrapper globalStruct;
extern void invalidateGlobals();

void testGlobalInvalidation() {
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}

  if (globalInt != 42)
    return;
  if (globalStruct.value != 43)
    return;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}

  // Repeat to make sure we don't get the /same/ new symbolic values.
  if (globalInt != 42)
    return;
  if (globalStruct.value != 43)
    return;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}
}

void testGlobalInvalidationWithDirectBinding() {
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}

  globalInt = 42;
  globalStruct.value = 43;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}
}

void testStaticLocals(void) {
  static int i;
  int tmp;

  extern int someSymbolicValue();
  i = someSymbolicValue();

  if (i == 5) {
    clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
    scanf("%d", &tmp);
    clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
    invalidateGlobals();
    clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
  }

  i = 6;
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  scanf("%d", &tmp);
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  invalidateGlobals();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}

  i = someSymbolicValue();
  if (i == 7) {
    clang_analyzer_eval(i == 7); // expected-warning{{TRUE}}
    scanf("%d", &i);
    clang_analyzer_eval(i == 7); // expected-warning{{UNKNOWN}}
  }

  i = 8;
  clang_analyzer_eval(i == 8); // expected-warning{{TRUE}}
  scanf("%d", &i);
  clang_analyzer_eval(i == 8); // expected-warning{{UNKNOWN}}
}

void testNonSystemGlobals(void) {
  extern int i;
  int tmp;

  if (i == 5) {
    clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
    scanf("%d", &tmp);
    clang_analyzer_eval(i == 5); // expected-warning{{TRUE}}
    invalidateGlobals();
    clang_analyzer_eval(i == 5); // expected-warning{{UNKNOWN}}
  }

  i = 6;
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  scanf("%d", &tmp);
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
  invalidateGlobals();
  clang_analyzer_eval(i == 6); // expected-warning{{UNKNOWN}}

  if (i == 7) {
    clang_analyzer_eval(i == 7); // expected-warning{{TRUE}}
    scanf("%d", &i);
    clang_analyzer_eval(i == 7); // expected-warning{{UNKNOWN}}
  }

  i = 8;
  clang_analyzer_eval(i == 8); // expected-warning{{TRUE}}
  scanf("%d", &i);
  clang_analyzer_eval(i == 8); // expected-warning{{UNKNOWN}}
}

void testWrappedGlobals(void) {
  extern char c;
  SomeStruct s;

  if (c == 'C') {
    s.p = &c;
    clang_analyzer_eval(c == 'C'); // expected-warning{{TRUE}}
    fakeSystemHeaderCall(0);
    clang_analyzer_eval(c == 'C'); // expected-warning{{TRUE}}
    fakeSystemHeaderCall(&s);
    clang_analyzer_eval(c == 'C'); // expected-warning{{UNKNOWN}}
  }

  c = 'c';
  s.p = &c;
  clang_analyzer_eval(c == 'c'); // expected-warning{{TRUE}}
  fakeSystemHeaderCall(0);
  clang_analyzer_eval(c == 'c'); // expected-warning{{TRUE}}
  fakeSystemHeaderCall(&s);
  clang_analyzer_eval(c == 'c'); // expected-warning{{UNKNOWN}}

  if (c == 'C') {
    s.p = &c;
    clang_analyzer_eval(c == 'C'); // expected-warning{{TRUE}}
    fakeSystemHeaderCall(0);
    clang_analyzer_eval(c == 'C'); // expected-warning{{TRUE}}
    fakeSystemHeaderCall(&s);
    clang_analyzer_eval(c == 'C'); // expected-warning{{UNKNOWN}}
  }
}

void testWrappedStaticsViaGlobal(void) {
  static char c;
  extern SomeStruct s;

  extern char getSomeChar();
  c = getSomeChar();

  if (c == 'C') {
    s.p = &c;
    clang_analyzer_eval(c == 'C'); // expected-warning{{TRUE}}
    invalidateGlobals();
    clang_analyzer_eval(c == 'C'); // expected-warning{{UNKNOWN}}
  }

  c = 'c';
  s.p = &c;
  clang_analyzer_eval(c == 'c'); // expected-warning{{TRUE}}
  invalidateGlobals();
  clang_analyzer_eval(c == 'c'); // expected-warning{{UNKNOWN}}
}
