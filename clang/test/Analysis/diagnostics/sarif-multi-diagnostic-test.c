// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.security.taint,debug.TaintTest,unix.Malloc %s -verify -analyzer-output=sarif -o - | %normalize_sarif | diff -U1 -b %S/Inputs/expected-sarif/sarif-multi-diagnostic-test.c.sarif -
#include "../Inputs/system-header-simulator.h"
#include "../Inputs/system-header-simulator-for-malloc.h"
#define ERR -1

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

int leak(int i) {
  void *mem = malloc(8);
  if (i < 4)
    return ERR; // expected-warning {{Potential leak of memory pointed to by 'mem'}}
  free(mem);
  return 0;
}

int unicode(void) {
  int løçål = 0;
  /* ☃ */ return 1 / løçål; // expected-warning {{Division by zero}}
}

int main(void) {
  f();
  g();
  h(0);
  leak(0);
  unicode();
  return 0;
}

