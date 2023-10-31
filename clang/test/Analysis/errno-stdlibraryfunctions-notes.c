// RUN: %clang_analyze_cc1 -verify -analyzer-output text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.Errno \
// RUN:   -analyzer-checker=alpha.unix.Errno \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true

#include "Inputs/errno_var.h"

int access(const char *path, int amode);

void clang_analyzer_warnIfReached();

void test1() {
  access("path", 0);
  access("path", 0);
  // expected-note@-1{{'errno' may be undefined after successful call to 'access'}}
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
    // expected-note@-2{{'errno' may be undefined after successful call to 'access'}}
    // expected-note@-3{{Taking true branch}}
    if (errno != 0) {
      // expected-warning@-1{{An undefined value may be read from 'errno'}}
      // expected-note@-2{{An undefined value may be read from 'errno'}}
    }
  }
}
