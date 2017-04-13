// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -analyzer-disable-all-checks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-disable-all-checks -analyzer-checker=core -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -analyzer-disable-all-checks -verify %s
// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -analyzer-disable-checker -verify %s 2>&1 | FileCheck %s
// expected-no-diagnostics

// CHECK: use -analyzer-disable-all-checks to disable all static analyzer checkers
int buggy() {
  int x = 0;
  return 5/x; // no warning
}
