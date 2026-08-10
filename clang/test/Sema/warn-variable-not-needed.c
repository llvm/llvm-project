// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// expected-no-diagnostics

static int a;
int bar(void) {
  extern int a;
  return a;
}
static int a;
