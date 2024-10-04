// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -verify %s

void foo(void) {
  // Should not crash at next line.
  int *p = (int*) 0x10000; // expected-warning{{Using a fixed address is not portable}}
  *p = 3;
}
