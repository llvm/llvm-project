// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-config alpha.unix.Errno:AllowErrnoReadOutsideConditionExpressions=false \
// RUN:   -DERRNO_VAR

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-config alpha.unix.Errno:AllowErrnoReadOutsideConditionExpressions=false \
// RUN:   -DERRNO_FUNC

#include "Inputs/system-header-simulator.h"
#ifdef ERRNO_VAR
#include "Inputs/errno_var.h"
#endif
#ifdef ERRNO_FUNC
#include "Inputs/errno_func.h"
#endif

int ErrnoTesterChecker_setErrnoIfError();

void test_cond() {
  ErrnoTesterChecker_setErrnoIfError();
  int A = errno ? 1 : 2;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_errno_store_into_variable() {
  ErrnoTesterChecker_setErrnoIfError();
  int a = errno;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

void test_errno_store_into_variable_in_expr() {
  ErrnoTesterChecker_setErrnoIfError();
  int a = 4 + errno;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

int test_errno_return() {
  ErrnoTesterChecker_setErrnoIfError();
  return errno;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}

int test_errno_return_expr() {
  ErrnoTesterChecker_setErrnoIfError();
  return errno > 10;
  // expected-warning@-1{{An undefined value may be read from 'errno'}}
}
