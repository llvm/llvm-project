// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s

const char a[4] = "abc";
void foo() {
  int i = 0;
  i = 1 > (a + 1, sizeof(a)); // expected-warning {{left operand of comma operator has no effect}}
}
