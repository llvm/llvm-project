// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker \
// RUN:   -analyzer-output=html -o %t %s
// RUN: ls %t | grep report | count 2

// Two parameters with identical spelling in different functions must
// not collide in the HTML issue hash: the enclosing function differs.

#include "mock-types.h"

RefCountable *provide_ref_cntbl();
void someFunction();

void foo(RefCountable* a) {
  a = provide_ref_cntbl();
  someFunction();
  a->method();
}

void baz(RefCountable* a) {
  a = provide_ref_cntbl();
  someFunction();
  a->method();
}
