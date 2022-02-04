// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

void test(void) {
  int z[1];
  __builtin_add_overflow(1, 1, z);
}
