// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker \
// RUN:   -analyzer-output=html -o %t %s
// RUN: ls %t | grep report | count 2

// Two local variables with identical spelling in different functions
// must not collide in the HTML issue hash: the enclosing function
// differs.

#include "mock-types.h"

void someFunction();

void foo() {
  RefCountable *bar;
  someFunction();
}

void baz() {
  RefCountable *bar;
  someFunction();
}
