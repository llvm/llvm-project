// RUN: %clang_cc1 -fsyntax-only -std=c99 -verify %s
// expected-no-diagnostics
void bb(int sz, int ar[sz][sz]) { }
