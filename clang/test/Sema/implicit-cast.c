// RUN: %clang_cc1 -fsyntax-only -verify %s

static char *test1(int cf) {
  return cf ? "abc" : 0;
}
static char *test2(int cf) {
  return cf ? 0 : "abc";
}

int baz(void) {
  int f;
  return ((void)0, f = 1.4f); // expected-warning {{implicit conversion from 'float' to 'int' changes value from 1.4 to 1}}
}
