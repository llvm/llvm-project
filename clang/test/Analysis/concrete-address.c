// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// expected-no-diagnostics

void foo(void) {
  int *p = (int*) 0x10000; // Should not crash here.
  *p = 3;
}
