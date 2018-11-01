// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.security.taint,debug.TaintTest %s -verify -analyzer-output=sarif -o - | diff -U1 -w -I ".*file:.*sarif-diagnostics-taint-test.c" -I "clang version" -I "2\.0\.0\-beta\." - %S/Inputs/expected-sarif/sarif-diagnostics-taint-test.c.sarif
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
