// RUN: %clang_analyze_cc1 -verify -analyzer-output text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -DERRNO_VAR

// RUN: %clang_analyze_cc1 -verify -analyzer-output text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=debug.ErrnoTest \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -DERRNO_FUNC

#include "Inputs/errno_var.h"
#include "Inputs/system-header-simulator.h"
#ifdef ERRNO_VAR
#include "Inputs/errno_var.h"
#endif
#ifdef ERRNO_FUNC
#include "Inputs/errno_func.h"
#endif

int ErrnoTesterChecker_setErrnoCheckState();

void something();

void testErrnoCheckUndefRead() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  something();
  X = ErrnoTesterChecker_setErrnoCheckState(); // expected-note{{Assuming that this function succeeds but sets 'errno' to an unspecified value}}
  if (X == 0) {                                // expected-note{{'X' is equal to 0}}
                                               // expected-note@-1{{Taking true branch}}
    if (errno) {
    } // expected-warning@-1{{An undefined value may be read from 'errno'}}
      // expected-note@-2{{An undefined value may be read from 'errno'}}
  }
}

void testErrnoCheckOverwrite() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  something();
  X = ErrnoTesterChecker_setErrnoCheckState(); // expected-note{{Assuming that this function returns 2. 'errno' should be checked to test for failure}}
  if (X == 2) {                                // expected-note{{'X' is equal to 2}}
                                               // expected-note@-1{{Taking true branch}}
    errno = 0;                                 // expected-warning{{Value of 'errno' was not checked and is overwritten here}}
                                               // expected-note@-1{{Value of 'errno' was not checked and is overwritten here}}
  }
}

void testErrnoCheckOverwriteStdCall() {
  int X = ErrnoTesterChecker_setErrnoCheckState();
  something();
  X = ErrnoTesterChecker_setErrnoCheckState(); // expected-note{{Assuming that this function returns 2. 'errno' should be checked to test for failure}}
  if (X == 2) {                                // expected-note{{'X' is equal to 2}}
                                               // expected-note@-1{{Taking true branch}}
    printf("");                                // expected-warning{{Value of 'errno' was not checked and may be overwritten by function 'printf'}}
                                               // expected-note@-1{{Value of 'errno' was not checked and may be overwritten by function 'printf'}}
  }
}
