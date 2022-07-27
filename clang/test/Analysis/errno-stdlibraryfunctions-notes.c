// RUN: %clang_analyze_cc1 -verify -analyzer-output text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true

#include "Inputs/errno_var.h"

int access(const char *path, int amode);

void clang_analyzer_warnIfReached();

void test1() {
  access("path", 0); // no note here
  access("path", 0);
  // expected-note@-1{{Assuming that function 'access' is successful, in this case the value 'errno' may be undefined after the call and should not be used}}
  if (errno != 0) {
    // expected-warning@-1{{An undefined value may be read from 'errno'}}
    // expected-note@-2{{An undefined value may be read from 'errno'}}
  }
}

void test2() {
  if (access("path", 0) == -1) {
    // expected-note@-1{{Taking true branch}}
    // Failure path.
    if (errno != 0) {
      // expected-note@-1{{'errno' is not equal to 0}}
      // expected-note@-2{{Taking true branch}}
      clang_analyzer_warnIfReached(); // expected-note {{REACHABLE}} expected-warning {{REACHABLE}}
    } else {
      clang_analyzer_warnIfReached(); // no-warning: We are on the failure path.
    }
  }
}

void test3() {
  if (access("path", 0) != -1) {
    // Success path.
    // expected-note@-2{{Assuming that function 'access' is successful, in this case the value 'errno' may be undefined after the call and should not be used}}
    // expected-note@-3{{Taking true branch}}
    if (errno != 0) {
      // expected-warning@-1{{An undefined value may be read from 'errno'}}
      // expected-note@-2{{An undefined value may be read from 'errno'}}
    }
  }
}
