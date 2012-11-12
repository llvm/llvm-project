// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// expected-no-diagnostics

void foo() {
  int *p = (int*) 0x10000; // Should not crash here.
  *p = 3;
}
