// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,debug.TaintTest %s -verify -analyzer-output=sarif -o - | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-diagnostics-taint-test.c.sarif -
#include "../Inputs/system-header-simulator.h"

int atoi(const char *nptr);

void f(void) {
  char s[80];
  scanf("%s", s);
  int d = atoi(s); // expected-warning {{tainted}}
}

int main(void) {
  f();
  return 0;
}
