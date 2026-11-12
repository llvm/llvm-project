// RUN: %clang_cc1 -fsyntax-only -verify %s 
// expected-no-diagnostics

void test() {
  int x;
  do
    int x;
  while (1);
}
