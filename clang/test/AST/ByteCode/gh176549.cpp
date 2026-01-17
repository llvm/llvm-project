// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// expected-no-diagnostics

const char a[4] = "abc";
void foo() {
  int i = 0;
  i = 1 > (a + 1, sizeof(a));
}
