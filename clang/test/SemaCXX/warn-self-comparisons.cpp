// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

void f(int (&array1)[2], int (&array2)[2]) {
  if (array1 == array2) { } // no warning
}
