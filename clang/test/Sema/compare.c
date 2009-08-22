// RUN: clang-cc -fsyntax-only -pedantic -verify %s

#include <stddef.h>

int test(char *C) { // nothing here should warn.
  return C != ((void*)0);
  return C != (void*)0;
  return C != 0;
}

int equal(char *a, const char *b) {
    return a == b;
}

int arrays(char (*a)[5], char(*b)[10], char(*c)[5]) {
  int d = (a == c);
  return a == b; // expected-warning {{comparison of distinct pointer types}}
}

int pointers(int *a) {
  return a > 0; // no warning.  rdar://7163039
  return a > (void *)0; // expected-warning {{comparison of distinct pointer types}}
}

int function_pointers(int (*a)(int), int (*b)(int)) {
  return a > b; // expected-warning {{ordered comparison of function pointers}}
  return function_pointers > function_pointers; // expected-warning {{ordered comparison of function pointers}}
  return a == (void *) 0;
  return a == (void *) 1; // expected-warning {{comparison of distinct pointer types}}
}

int void_pointers(void *foo) {
  return foo == NULL;
}
