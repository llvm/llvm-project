// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region \
// RUN:   -analyzer-disable-all-checks -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-disable-all-checks -analyzer-checker=core \
// RUN:   -analyzer-store=region -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-disable-all-checks -verify %s
//
// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region \
// RUN:   -analyzer-disable-checker non.existant.Checker -verify %s 2>&1 \
// RUN:   | FileCheck %s
//
// expected-no-diagnostics

// CHECK: no analyzer checkers or packages are associated with 'non.existant.Checker'
// CHECK: use -analyzer-disable-all-checks to disable all static analyzer checkers
int buggy(void) {
  int x = 0;
  return 5/x; // no warning
}
