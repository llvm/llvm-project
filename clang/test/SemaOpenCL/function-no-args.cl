// RUN: %clang_cc1 -verify -pedantic -fsyntax-only -cl-std=CL2.0 %s
// expected-no-diagnostics

global int gi;
int my_func();
int my_func() {
  gi = 2;
  return gi;
}
