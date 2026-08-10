// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -DERRNO_VAR

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -DERRNO_FUNC

#include "Inputs/system-header-simulator.h"
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
int ErrnoTesterChecker_setErrnoCheckState();

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

void testErrnoCheck0() {
  // If the function returns a success result code, value of 'errno'
  // is unspecified and it is unsafe to make any decision with it.
  // The function did not promise to not change 'errno' if no failure happens.
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    if (errno) { // expected-warning{{An undefined value may be read from 'errno' [unix.Errno]}}
    }
    if (errno) { // no warning for second time (analysis stops at the first warning)
    }
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    if (errno) { // expected-warning{{An undefined value may be read from 'errno' [unix.Errno]}}
    }
    errno = 0;
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    errno = 0;
    if (errno) { // no warning after overwritten 'errno'
    }
  }
}

void testErrnoCheck1() {
  // If the function returns error result code that is out-of-band (not a valid
  // non-error return value) the value of 'errno' can be checked but it is not
  // required to do so.
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 1) {
    if (errno) { // no warning
    }
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 1) {
    errno = 0; // no warning
  }
}

void testErrnoCheck2() {
  // If the function returns an in-band error result the value of 'errno' is
  // required to be checked to verify if error happened.
  // The same applies to other functions that can indicate failure only by
  // change of 'errno'.
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    errno = 0; // expected-warning{{Value of 'errno' was not checked and is overwritten here [unix.Errno]}}
    errno = 0;
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    errno = 0; // expected-warning{{Value of 'errno' was not checked and is overwritten here [unix.Errno]}}
    if (errno) {
    }
  }
}

void testErrnoCheck3() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    if (errno) {
    }
    errno = 0; // no warning after 'errno' was read
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    int A = errno;
    errno = 0; // no warning after 'errno' was read
  }
}

void testErrnoCheckUndefinedLoad() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    if (errno) { // expected-warning{{An undefined value may be read from 'errno' [unix.Errno]}}
    }
  }
}

void testErrnoNotCheckedAtSystemCall() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    printf("%i", 1); // expected-warning{{Value of 'errno' was not checked and may be overwritten by function 'printf' [unix.Errno]}}
    printf("%i", 1); // no warning ('printf' does not change errno state)
  }
}

void testErrnoCheckStateInvalidate() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    something();
    if (errno) { // no warning after an invalidating function call
    }
  }
  X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 0) {
    printf("%i", 1);
    if (errno) { // no warning after an invalidating standard function call
    }
  }
}

void testErrnoCheckStateInvalidate1() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  if (X == 2) {
    clang_analyzer_eval(errno); // expected-warning{{TRUE}}
    something();
    clang_analyzer_eval(errno); // expected-warning{{UNKNOWN}}
    errno = 0;                  // no warning after invalidation
  }
}

void test_if_cond_in_expr() {
  ErrnoTesterChecker_setErrnoIfError();
  if (errno + 10 > 2) {
    // expected-warning@-1{{An undefined value may be read from 'errno'}}
  }
}

void test_for_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  for (; errno != 0;) {
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
  }
}

void test_do_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  do {
  } while (errno != 0);
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_while_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  while (errno != 0) {
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
  }
}

void test_switch_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  switch (errno) {}
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_conditional_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  int A = errno ? 1 : 2;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_binary_conditional_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  int A = errno ?: 2;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_errno_store_into_variable() {
  ErrnoTesterChecker_setErrnoIfError();
  int a = errno; // AllowNonConditionErrnoRead is on by default, no warning
}

void test_errno_store_into_variable_in_expr() {
  ErrnoTesterChecker_setErrnoIfError();
  int a = errno > 1; // AllowNonConditionErrnoRead is on by default, no warning
}

int test_errno_return() {
  ErrnoTesterChecker_setErrnoIfError();
  return errno;
}

void test_errno_pointer1() {
  ErrnoTesterChecker_setErrnoIfError();
  int *ErrnoP = &errno;
  int A = errno ? 1 : 2;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_errno_pointer2() {
  ErrnoTesterChecker_setErrnoIfError();
  int *ErrnoP = &errno;
  int A = (*ErrnoP) ? 1 : 2;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

int f(int);

void test_errno_in_condition_in_function_call() {
  ErrnoTesterChecker_setErrnoIfError();
  if (f(errno) != 0) {
  }
}
