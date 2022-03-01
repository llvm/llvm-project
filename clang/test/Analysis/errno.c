// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -DERRNO_VAR

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -DERRNO_FUNC

#ifdef ERRNO_VAR
#include "Inputs/errno_var.h"
#endif
#ifdef ERRNO_FUNC
#include "Inputs/errno_func.h"
#endif

void clang_analyzer_eval(int);
void ErrnoTesterChecker_setErrno(int);
int ErrnoTesterChecker_getErrno();
int ErrnoTesterChecker_setErrnoIfError();
int ErrnoTesterChecker_setErrnoIfErrorRange();

void something();

void test() {
  // Test if errno is initialized.
  clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}

  ErrnoTesterChecker_setErrno(1);
  // Test if errno was recognized and changed.
  clang_analyzer_eval(errno == 1);                         // expected-warning{{TRUE}}
  clang_analyzer_eval(ErrnoTesterChecker_getErrno() == 1); // expected-warning{{TRUE}}

  something();

  // Test if errno was invalidated.
  clang_analyzer_eval(errno);                         // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(ErrnoTesterChecker_getErrno()); // expected-warning{{UNKNOWN}}
}

void testRange(int X) {
  if (X > 0) {
    ErrnoTesterChecker_setErrno(X);
    clang_analyzer_eval(errno > 0); // expected-warning{{TRUE}}
  }
}

void testIfError() {
  if (ErrnoTesterChecker_setErrnoIfError())
    clang_analyzer_eval(errno == 11); // expected-warning{{TRUE}}
}

void testIfErrorRange() {
  if (ErrnoTesterChecker_setErrnoIfErrorRange()) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(errno == 1); // expected-warning{{FALSE}} expected-warning{{TRUE}}
  }
}
