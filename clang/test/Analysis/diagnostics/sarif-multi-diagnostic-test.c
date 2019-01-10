// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.security.taint,debug.TaintTest %s -verify -analyzer-output=sarif -o - | %diff_sarif %S/Inputs/expected-sarif/sarif-multi-diagnostic-test.c.sarif -
#include "../Inputs/system-header-simulator.h"

int atoi(const char *nptr);

void f(void) {
  char s[80];
  scanf("%s", s);
  int d = atoi(s); // expected-warning {{tainted}}
}

void g(void) {
  void (*fp)(int);
  fp(12); // expected-warning {{Called function pointer is an uninitialized pointer value}}
}

int h(int i) {
  if (i == 0)
    return 1 / i; // expected-warning {{Division by zero}}
  return 0;
}

int main(void) {
  f();
  g();
  h(0);
  return 0;
}

